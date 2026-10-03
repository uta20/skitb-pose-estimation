"""
`run_pose_estimation.py` のOutput``keypoints.csv`` を読み込み
``src/features.py`` で力学的特徴量(体幹の折れ角・脚のV字角・膝伸展角・
左右非対称度)を計算して ``features.csv`` として保存するスクリプト

使い方:
    # 事前に run_pose_estimation.py で keypoints.csv を生成
    uv run scripts/run_pose_estimation.py --sequence JP0001

    # keypoints.csv から特徴量を計算
    uv run scripts/extract_features.py --sequence JP0001

    # 複数/全シーケンスもまとめて処理可能
    uv run scripts/extract_features.py --sequence JP0001,JP0002
    uv run scripts/extract_features.py --sequence all

出力:
    outputs/<sequence>/features.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.dataset import discover_sequences
from src.features import extract_features
from src.pose import COCO_KEYPOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "trunk_hip_angle_deg",
    "leg_v_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "knee_asymmetry_deg",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", required=True,
        help="シーケンスID(例: JP0001). カンマ区切りで複数, 'all' で全シーケンス",
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs"),
        help="run_pose_estimation.py の出力先と同じディレクトリ (デフォルト: ./outputs)",
    )
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "data"),
        help="'--sequence all' 使用時にシーケンス一覧を検出するディレクトリ",
    )
    return parser.parse_args()

def resolve_sequence_ids(args: argparse.Namespace) -> list[str]:
    if args.sequence.lower() == "all":
        return discover_sequences(Path(args.data_root), category="JP")
    return [s.strip() for s in args.sequence.split(",") if s.strip()]

def parse_keypoints_row(row: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    """keypoints.csvの1行(dict)を、extract_features()が受け取る形式に変換する。"""
    keypoints: dict[str, tuple[float, float, float]] = {}
    for name in COCO_KEYPOINT_NAMES:
        x_str, y_str, c_str = row.get(f"{name}_x"), row.get(f"{name}_y"), row.get(f"{name}_conf")
        if not x_str or not y_str or not c_str:
            continue  # 未検出フレームはスキップ
        keypoints[name] = (float(x_str), float(y_str), float(c_str))
    return keypoints

def process_one_sequence(sequence_id: str, out_dir: Path) -> int:
    """1シーケンス分のkeypoints.csvを読み, features.csvを書き出し, 処理行数を返す"""
    keypoints_csv = out_dir / sequence_id / "keypoints.csv"
    if not keypoints_csv.exists():
        print(f"[skip] {keypoints_csv} が見つかりません(先にrun_pose_estimation.pyを実行してください)")
        return 0

    features_csv = out_dir / sequence_id / "features.csv"
    n_rows = 0
    with keypoints_csv.open("r", encoding="utf-8") as f_in, \
         features_csv.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["frame_id", "detected", *FEATURE_COLUMNS])

        for row in reader:
            n_rows += 1
            detected = row.get("detected") == "True"
            if not detected:
                writer.writerow([row["frame_id"], False, *([""] * len(FEATURE_COLUMNS))])
                continue

            keypoints = parse_keypoints_row(row)
            features = extract_features(keypoints)
            writer.writerow([
                row["frame_id"], True,
                *(getattr(features, col) for col in FEATURE_COLUMNS),
            ])

    print(f"[{sequence_id}] {n_rows}行を処理 -> {features_csv}")
    return n_rows

def main() -> None:
    args = parse_args()
    sequence_ids = resolve_sequence_ids(args)
    if not sequence_ids:
        raise SystemExit("処理対象のシーケンスが見つかりませんでした。")

    out_dir = Path(args.out_dir)
    total = 0
    for sequence_id in sequence_ids:
        total += process_one_sequence(sequence_id, out_dir)
    print(f"\n合計 {total} フレーム分の特徴量を計算しました。")


if __name__ == "__main__":
    main()