"""
指定したSkiTBシーケンスに対して姿勢推定を実行するCLI

シーケンスの選び方(1件/カンマ区切り/全件)を解釈し
``src/pipeline.py`` の ``process_sequence()`` をループする

使い方:
    # 単一シーケンス、利用可能な全フレームを処理
    uv run scripts/run_pose_estimation.py --sequence JP0001

    # 単一フレームのみ
    uv run scripts/run_pose_estimation.py --sequence JP0001 --frame 63

    # 複数シーケンスをカンマ区切りで指定
    uv run scripts/run_pose_estimation.py --sequence JP0001,JP0002,JP0003

    # data/ 直下で検出できる全シーケンス(JP0001〜JP0100)を処理
    uv run scripts/run_pose_estimation.py --sequence all

    # 全件処理だが、動作確認のため先頭5シーケンスだけに絞る
    uv run scripts/run_pose_estimation.py --sequence all --limit 5

    # 可視化画像の保存を省略し、CSV出力のみ実行
    uv run scripts/run_pose_estimation.py --sequence all --no-images

出力:
    outputs/<sequence>/<frame_id>_pose.jpg   キーポイント可視化画像 (--no-images指定時は省略)
    outputs/<sequence>/keypoints.csv         各シーケンスのキーポイント座標
    outputs/summary.csv                      全シーケンスの処理結果サマリー(検出率など)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from src.dataset import discover_sequences
from src.pipeline import SequenceResult, process_sequence
from src.pose import PoseEstimator

REPO_ROOT = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    # コマンドライン引数の定義, パース
    parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        "--sequence", required=True,
        help="シーケンスID(JP0XXX) カンマ区切りで複数指定"
             "または 'all' でdata/直下の全シーケンスを対象にできる",
    )
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "data"),
        help="シーケンスフォルダの親ディレクトリ (デフォルト: ./data)",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="処理する単一フレーム番号. '--sequence all/カンマ区切り' と併用不可。"
             "省略時は各シーケンスの利用可能な全フレームを処理",
    )
    parser.add_argument(
        "--model", default="yolov8n-pose.pt",
        help="Ultralytics yolov8n-pose.pt",
    )
    parser.add_argument(
        "--margin", type=float, default=0.25,
        help="BBoxクロップ時のマージン比率",
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs"),
        help="出力先ディレクトリ",
    )
    parser.add_argument(
            "--limit", type=int, default=None,
            help="処理するシーケンス数の上限(動作確認用。'--sequence all'と併用)",
        )
    parser.add_argument(
            "--no-images", action="store_true",
            help="キーポイント可視化画像保存を省略、CSV出力のみ行う"
                 "(大量シーケンスのバッチ処理を高速化したい場合)",
        )
    return parser.parse_args()

def resolve_sequence_ids(args: argparse.Namespace) -> list[str]:
    """--sequence 引数(単一/カンマ区切り/all)から実際のシーケンスID一覧を返す"""
    data_root = Path(args.data_root)

    if args.sequence.lower() == "all":
        sequence_ids = discover_sequences(data_root, category="JP")
        if args.limit is not None:
            sequence_ids = sequence_ids[: args.limit]
        return sequence_ids

    sequence_ids = [s.strip() for s in args.sequence.split(",") if s.strip()]
    if args.frame is not None and len(sequence_ids) > 1:
        raise SystemExit("--frame は単一シーケンス指定時のみ使用可能")
    return sequence_ids

def write_summary(results: list[SequenceResult], out_dir: Path) -> Path:
    summary_path = Path(out_dir) / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sequence_id", "frames_total", "frames_detected", "detection_rate", "error"]
        )
        for r in results:
            writer.writerow([
                r.sequence_id, r.frames_total, r.frames_detected,
                f"{r.detection_rate:.3f}", r.error or "",
            ])
    return summary_path

def main() -> None:
    args = parse_args()

    sequence_ids = resolve_sequence_ids(args)
    if not sequence_ids:
        raise SystemExit(
            f"{args.data_root} 配下に処理対象のシーケンスが見つかりませんでした"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデルのロードは重いため
    # 全シーケンスを通じて1度だけ行い、process_sequence()へ使い回す
    estimator = PoseEstimator(model_name=args.model)

    frame_ids = [args.frame] if args.frame is not None else None

    results: list[SequenceResult] = []
    for sequence_id in tqdm(sequence_ids, desc="sequences", unit="seq"):
        result = process_sequence(
            sequence_id=sequence_id,
            data_root=args.data_root,
            out_dir=out_dir,
            estimator=estimator,
            margin=args.margin,
            frame_ids=frame_ids,
            save_images=not args.no_images,
        )
        results.append(result)

        if result.error:
            tqdm.write(f"[{sequence_id}] スキップ: {result.error}")
        else:
            tqdm.write(
                f"[{sequence_id}] {result.frames_detected}/{result.frames_total} "
                f"フレームで検出 (検出率 {result.detection_rate:.1%})"
                f" -> {result.keypoints_csv}"
            )

    # 複数シーケンスを処理した場合のみサマリーCSVを出力する
    if len(results) > 1:
        summary_path = write_summary(results, out_dir)
        succeeded = sum(1 for r in results if r.error is None)
        print(
            f"\n{succeeded}/{len(results)} シーケンスを処理しました。"
            f"サマリーを出力: {summary_path}"
        )


if __name__ == "__main__":
    main()