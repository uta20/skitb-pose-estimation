"""v0.5: SkiTBの可視性ラベル(visibilities.txt)を手がかりに、
検出失敗率とキーポイント信頼度の分布を定量化する評価モジュール。

## 「可視性ラベル」を正解として使うことの限界

SkiTBの ``visibilities.txt`` は「フレームごとに選手の50%以上が画像内に
可視かどうか」を表す0/1ラベルであり、姿勢推定の正解(ground truth)
として設計されたものではない。したがって本モジュールが計算する指標には
以下の近似が含まれる。

- 可視ラベルが1でも、選手が極端に小さい・ブレている等の理由で検出が
  失敗することがある(これはモデルの能力的な限界であり、ラベルの誤りではない)
- 可視ラベルが0(50%以上遮蔽)でも、実際には検出できることがある
  (この場合の検出成功は、本モジュールの集計上は便宜的に「FP」として
  扱われるが、モデルが悪いわけではなく、むしろ頑健性の表れでありうる)

そのため、本モジュールの出力は**モデルの正解率を厳密に表すものではなく、
「可視性ラベルと検出結果の一致度」という近似指標**として解釈すること。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .pose import COCO_KEYPOINT_NAMES


@dataclass
class ConfusionCounts:
    """可視性ラベル vs 検出結果の混同行列(1シーケンス分、または全体集計分)。

    - tp: 可視=1, 検出=True  (正しく検出)
    - fn: 可視=1, 検出=False (見逃し。最も問題視すべき失敗パターン)
    - fp: 可視=0, 検出=True  (50%以上遮蔽でも検出。上記の限界を参照)
    - tn: 可視=0, 検出=False (遮蔽時に検出しなかった、という一致)
    """

    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        """可視ラベル=1のフレームのうち、実際に検出できた割合。
        「見逃し率」は 1 - recall で求められる。
        """
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.total if self.total else None

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            tp=self.tp + other.tp, fn=self.fn + other.fn,
            fp=self.fp + other.fp, tn=self.tn + other.tn,
        )


def compute_confusion(
    frame_visibility: dict[int, bool],
    frame_detected: dict[int, bool],
) -> ConfusionCounts:
    """frame_idをキーにした可視性・検出結果の辞書から混同行列を計算する。

    両方の辞書に共通して存在するframe_idのみを対象とする
    (片方にしか無いframe_idは、対応漏れとして無視する)。
    """
    counts = ConfusionCounts()
    common_ids = set(frame_visibility) & set(frame_detected)
    for frame_id in common_ids:
        visible = frame_visibility[frame_id]
        detected = frame_detected[frame_id]
        if visible and detected:
            counts.tp += 1
        elif visible and not detected:
            counts.fn += 1
        elif not visible and detected:
            counts.fp += 1
        else:
            counts.tn += 1
    return counts


def load_keypoints_csv(path: Path) -> pd.DataFrame:
    """run_pose_estimation.pyが出力する keypoints.csv を読み込む。

    ``detected`` 列はCSV上は文字列 "True"/"False" として保存されているため、
    bool型に変換してから返す。
    """
    df = pd.read_csv(path)
    df["detected"] = df["detected"].astype(str).str.lower().isin(["true", "1"])
    return df


def keypoint_confidence_stats(keypoints_df: pd.DataFrame) -> pd.DataFrame:
    """検出できたフレームのみを対象に、関節ごとの信頼度(conf)の
    分布(平均・中央値・標準偏差・下位10%点)を集計する。

    検出に失敗したフレーム(detected=False)は、そもそもconf値が
    存在しないため集計対象から除外する。
    """
    detected = keypoints_df[keypoints_df["detected"]]
    rows = []
    for name in COCO_KEYPOINT_NAMES:
        col = f"{name}_conf"
        if col not in detected.columns:
            continue
        series = pd.to_numeric(detected[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append({
            "keypoint": name,
            "mean_conf": series.mean(),
            "median_conf": series.median(),
            "std_conf": series.std(),
            "p10_conf": series.quantile(0.1),
            "n_frames": len(series),
        })
    return pd.DataFrame(rows).sort_values("mean_conf")


def write_confusion_csv(path: Path, rows: list[tuple[str, ConfusionCounts]]) -> None:
    """``[(sequence_id, ConfusionCounts), ...]`` をCSVに書き出す。"""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_id", "tp", "fn", "fp", "tn", "total",
            "precision", "recall", "accuracy",
        ])
        for sequence_id, c in rows:
            writer.writerow([
                sequence_id, c.tp, c.fn, c.fp, c.tn, c.total,
                f"{c.precision:.4f}" if c.precision is not None else "",
                f"{c.recall:.4f}" if c.recall is not None else "",
                f"{c.accuracy:.4f}" if c.accuracy is not None else "",
            ])