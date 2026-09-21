"""
1シーケンス分の姿勢推定処理をまとめたパイプライン

``scripts/run_pose_estimation.py`` から処理ロジック本体をこちらへ切り出し
シーケンス数に関わらず関数を再利用できるように構築
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import cv2
from .crop import crop_image, expand_box
from .dataset import SkiSequence
from .pose import COCO_KEYPOINT_NAMES, PoseEstimator
from .visualize import draw_pose


@dataclass
class SequenceResult:
    """1シーケンス分の処理結果サマリー。"""

    sequence_id: str
    frames_total: int      # 処理対象となったフレーム数(画像が存在するもの)
    frames_detected: int   # 選手を検出できたフレーム数
    keypoints_csv: Path | None
    error: str | None = None

    @property
    def detection_rate(self) -> float:
        """検出率 (0.0〜1.0)。処理対象が0件の場合は0.0を返す。"""
        if self.frames_total == 0:
            return 0.0
        return self.frames_detected / self.frames_total


def process_sequence(
    sequence_id: str,
    data_root: Path,
    out_dir: Path,
    estimator: PoseEstimator | None,
    margin: float = 0.25,
    frame_ids: list[int] | None = None,
    save_images: bool = True,
) -> SequenceResult:
    """1シーケンス分の姿勢推定を実行し、結果を ``<out_dir>/<sequence_id>/`` に保存する。

    Parameters
    ----------
    sequence_id:
        シーケンスID (例: "JP0001")。
    data_root:
        シーケンスフォルダの親ディレクトリ (通常は ``data/``)。
    out_dir:
        出力先の親ディレクトリ (通常は ``outputs/``)。
    estimator:
        姿勢推定器。処理対象フレームが1件も無い場合は使用されないため
        ``None`` を渡してもよい(呼び出し元でモデルロードを遅延させたい
        場合に利用する)。
    margin:
        BBoxクロップ時のマージン比率。
    frame_ids:
        処理対象のフレームID一覧。``None`` の場合は
        ``SkiSequence.available_frame_ids()`` (実際に画像が存在するフレーム)
        を自動的に対象とする。
    save_images:
        Trueならキーポイント可視化画像も保存する。バッチ処理で大量の
        シーケンスを回す際、ディスク容量・処理時間を抑えたい場合は
        ``False`` にしてCSV出力のみに絞ることができる。
    """
    sequence_dir = Path(data_root) / sequence_id

    try:
        sequence = SkiSequence.from_mc_dir(sequence_dir)
    except (FileNotFoundError, ValueError) as e:
        return SequenceResult(sequence_id, 0, 0, None, error=f"アノテーション読み込み失敗: {e}")

    target_ids = frame_ids if frame_ids is not None else sequence.available_frame_ids()
    if not target_ids:
        return SequenceResult(
            sequence_id, 0, 0, None,
            error=f"{sequence_dir}/frames に処理可能な画像がありません",
        )

    seq_out_dir = Path(out_dir) / sequence_id
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = seq_out_dir / "keypoints.csv"

    detected = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_id", "detected"] + [
                f"{name}_{axis}" for name in COCO_KEYPOINT_NAMES for axis in ("x", "y", "conf")
            ]
        )

        for frame_id in sorted(target_ids):
            ann = sequence.get(frame_id)
            image_path = sequence.frame_path(frame_id)
            image = cv2.imread(str(image_path))
            if image is None:
                writer.writerow([frame_id, False] + [""] * (len(COCO_KEYPOINT_NAMES) * 3))
                continue

            h, w = image.shape[:2]
            box = expand_box(ann.xyxy, (w, h), margin_ratio=margin)
            crop = crop_image(image, box)

            pose = estimator.predict(crop, offset_xy=(box[0], box[1]))

            if pose is None:
                writer.writerow([frame_id, False] + [""] * (len(COCO_KEYPOINT_NAMES) * 3))
                continue

            detected += 1
            if save_images:
                vis = draw_pose(image, pose, box_xyxy=ann.xyxy)
                cv2.imwrite(str(seq_out_dir / f"{frame_id:05d}_pose.jpg"), vis)

            row = [frame_id, True]
            for (x, y), c in zip(pose.keypoints_xy, pose.keypoints_conf):
                row.extend([f"{x:.1f}", f"{y:.1f}", f"{c:.3f}"])
            writer.writerow(row)

    return SequenceResult(sequence_id, len(target_ids), detected, csv_path)