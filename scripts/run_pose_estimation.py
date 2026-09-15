"""
指定したSkiTBシーケンスに対してクロップと姿勢推定を実行し, 
CSVと可視化画像を生成するCLIスクリプト

v0.1では JP0001 の同梱サンプルフレーム(1枚)のみで動作確認できる状態だが, 
`data/JPxxxx/frames/` に全フレーム画像を配置すれば
全シーケンスに拡張できる設計

使い方:
    uv run scripts/run_pose_estimation.py --sequence JP0001
    uv run scripts/run_pose_estimation.py --sequence JP0001 --frame 63

出力:
    outputs/<sequence>/<frame_id>_pose.jpg   キーポイント可視化画像
    outputs/<sequence>/keypoints.csv         全フレームのキーポイント座標
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
from src.crop import crop_image, expand_box
from src.dataset import SkiSequence
from src.pose import COCO_KEYPOINT_NAMES, PoseEstimator
from src.visualize import draw_pose

REPO_ROOT = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    # コマンドライン引数の定義, パース
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", required=True,
        help="シーケンスID (例: JP0001)",
    )
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "data"),
        help="シーケンスフォルダの親ディレクトリ (デフォルト: ./data)",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="処理対象フレーム番号. 省略時は利用可能な全フレームを処理",
    )
    parser.add_argument(
        "--model", default="yolov8n-pose.pt",
        help="Ultralytics YOLOv8-poseモデル名 (デフォルト: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.25,
        help="BBoxクロップ時のマージン比率",
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs"),
        help="出力先ディレクトリ",
    )
    return parser.parse_args()

def main() -> None:
    # 前処理, 推論, 可視化, I/Oを統括するループ
    args = parse_args()

    sequence_dir = Path(args.data_root) / args.sequence
    sequence = SkiSequence.from_mc_dir(sequence_dir)
    # 処理対象フレームIDの決定
    frame_ids = [args.frame] if args.frame is not None else sequence.available_frame_ids()
    if not frame_ids:
        raise SystemExit(
            f"{sequence_dir}/frames に処理可能なフレーム画像が見つかりません。"
        )

    out_dir = Path(args.out_dir) / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = PoseEstimator(model_name=args.model)
    # COCOキーポイントのヘッダを含むCSVを作成
    csv_path = out_dir / "keypoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_id", "detected"] + [
                f"{name}_{axis}" for name in COCO_KEYPOINT_NAMES for axis in ("x", "y", "conf")
            ]
        )

        for frame_id in sorted(frame_ids):
            # アノテーションと画像取得
            ann = sequence.get(frame_id)
            image_path = sequence.frame_path(frame_id)
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[skip] 画像が読み込めません: {image_path}")
                continue
            # BBoxを拡張してクロップ
            h, w = image.shape[:2]
            box = expand_box(ann.xyxy, (w, h), margin_ratio=args.margin)
            crop = crop_image(image, box)
            # 姿勢推定
            pose = estimator.predict(crop, offset_xy=(box[0], box[1]))

            if pose is None:
                print(f"[frame {frame_id}] 選手を検出できませんでした")
                writer.writerow([frame_id, False] + [""] * (len(COCO_KEYPOINT_NAMES) * 3))
                continue
            # 可視化画像の生成
            vis = draw_pose(image, pose, box_xyxy=ann.xyxy)
            out_path = out_dir / f"{frame_id:05d}_pose.jpg"
            cv2.imwrite(str(out_path), vis)
            # CSVにキーポイント座標を出力
            row = [frame_id, True]
            for (x, y), c in zip(pose.keypoints_xy, pose.keypoints_conf):
                row.extend([f"{x:.1f}", f"{y:.1f}", f"{c:.3f}"])
            writer.writerow(row)

            print(f"[frame {frame_id}] 検出conf={pose.bbox_conf:.2f} -> {out_path}")

    print(f"\nキーポイントCSVを出力しました: {csv_path}")


if __name__ == "__main__":
    main()