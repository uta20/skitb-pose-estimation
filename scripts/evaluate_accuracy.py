"""v0.5: SkiTBの可視性ラベルを手がかりに検出失敗率を定量化し、
キーポイントごとの信頼度分布を集計するCLI。

**前提**: 対象シーケンスについて、先に
    uv run scripts/run_pose_estimation.py --sequence <...>
を実行し、``outputs/<sequence>/keypoints.csv`` が生成済みであること
(本スクリプトは姿勢推定自体は行わない)。

使い方:
    uv run scripts/evaluate_accuracy.py --sequence JP0001
    uv run scripts/evaluate_accuracy.py --sequence JP0001,JP0002
    uv run scripts/evaluate_accuracy.py --sequence all

出力:
    outputs/evaluation_summary.csv   シーケンス別の混同行列・precision/recall/accuracy
    outputs/keypoint_confidence.csv  関節別の信頼度分布(対象シーケンス全体で合算)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from src.dataset import SkiSequence, discover_sequences

from src.evaluation import (
    ConfusionCounts,
    compute_confusion,
    keypoint_confidence_stats,
    load_keypoints_csv,
    write_confusion_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sequence", required=True,
        help="シーケンスID。単一/カンマ区切り/'all'(data/直下の全検出シーケンス)",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def resolve_sequence_ids(args: argparse.Namespace) -> list[str]:
    data_root = Path(args.data_root)
    if args.sequence.lower() == "all":
        return discover_sequences(data_root, category="JP")
    return [s.strip() for s in args.sequence.split(",") if s.strip()]


def main() -> None:
    args = parse_args()
    sequence_ids = resolve_sequence_ids(args)
    out_dir = Path(args.out_dir)

    confusion_rows: list[tuple[str, ConfusionCounts]] = []
    all_keypoints_dfs: list[pd.DataFrame] = []

    for sequence_id in sequence_ids:
        keypoints_path = out_dir / sequence_id / "keypoints.csv"
        if not keypoints_path.exists():
            print(
                f"[{sequence_id}] スキップ: {keypoints_path} が見つかりません"
                f"(先に run_pose_estimation.py を実行してください)"
            )
            continue

        try:
            sequence = SkiSequence.from_mc_dir(Path(args.data_root) / sequence_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{sequence_id}] スキップ: アノテーション読み込み失敗 ({e})")
            continue

        kp_df = load_keypoints_csv(keypoints_path)
        all_keypoints_dfs.append(kp_df)

        frame_visibility = {ann.frame_id: ann.visible for ann in sequence}
        frame_detected = dict(zip(kp_df["frame_id"], kp_df["detected"], strict=False))

        counts = compute_confusion(frame_visibility, frame_detected)
        confusion_rows.append((sequence_id, counts))

        recall_str = f"{counts.recall:.1%}" if counts.recall is not None else "N/A"
        print(
            f"[{sequence_id}] 可視フレームの検出率(recall)={recall_str} "
            f"(TP={counts.tp}, FN={counts.fn}, FP={counts.fp}, TN={counts.tn})"
        )

    if not confusion_rows:
        raise SystemExit("評価可能なシーケンスがありませんでした。")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "evaluation_summary.csv"
    write_confusion_csv(summary_path, confusion_rows)

    overall = sum((c for _, c in confusion_rows), ConfusionCounts())
    recall_str = f"{overall.recall:.1%}" if overall.recall is not None else "N/A"
    precision_str = (
        f"{overall.precision:.1%}" if overall.precision is not None else "N/A"
    )
    accuracy_str = (
        f"{overall.accuracy:.1%}" if overall.accuracy is not None else "N/A"
    )
    print(
        f"\n全体: recall(見逃し率の逆)={recall_str} precision={precision_str} "
        f"accuracy={accuracy_str} (n={overall.total}フレーム)"
    )
    print(f"[saved] {summary_path}")

    if all_keypoints_dfs:
        merged = pd.concat(all_keypoints_dfs, ignore_index=True)
        conf_stats = keypoint_confidence_stats(merged)
        conf_path = out_dir / "keypoint_confidence.csv"
        conf_stats.to_csv(conf_path, index=False)
        print(f"[saved] {conf_path}")
        print("\n信頼度が低い関節トップ5:")
        print(conf_stats.head(5).to_string(index=False))


if __name__ == "__main__":
    main()