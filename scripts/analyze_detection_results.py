"""姿勢推定の検出率(outputs/summary.csv)と、メタデータ・視覚属性
(outputs/manifest.csv, data/JP_visual_attributes.csv)を突き合わせ、
「どういうシーケンスで検出が落ちるか」を分析するスクリプト。

前提: 先に以下を実行して両ファイルを生成しておくこと。
    uv run scripts/build_dataset_manifest.py
    uv run scripts/run_pose_estimation.py --sequence all

やっていること:
1. summary.csv (検出率) と manifest.csv (メタデータ+split) を結合
2. 視覚属性 (FOC/POC/MB等, dateスプリットのtest集合のみ) を結合
3. 各視覚属性の有無で検出率の平均に差があるかを比較
4. 天候(Weather)・会場(HillLocation)別の平均検出率を集計
5. 検出率が低い順にシーケンスを一覧表示(定性的に見に行く対象の選定用)

使い方:
    uv run scripts/analyze_detection_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metadata import VISUAL_ATTRIBUTE_COLUMNS, load_visual_attributes

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"


def main() -> None:
    summary_path = OUT_DIR / "summary.csv"
    manifest_path = OUT_DIR / "manifest.csv"

    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} が見つかりません。先に "
            "'uv run scripts/run_pose_estimation.py --sequence all' を実行してください。"
        )
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} が見つかりません。先に "
            "'uv run scripts/build_dataset_manifest.py' を実行してください。"
        )

    summary = pd.read_csv(summary_path, index_col="sequence_id")
    manifest = pd.read_csv(manifest_path, index_col=0)

    merged = manifest.join(summary[["frames_total", "frames_detected", "detection_rate"]])
    merged = merged.dropna(subset=["detection_rate"])
    print(f"[結合結果] {len(merged)} シーケンス分の検出率とメタデータを結合しました\n")

    # 1. 視覚属性ごとの平均検出率(dateスプリットのtest集合のみ存在)
    visual_path = DATA_DIR / "JP_visual_attributes.csv"
    if visual_path.exists():
        visual = load_visual_attributes(visual_path)
        visual_seq = visual.groupby("sequence_id")[VISUAL_ATTRIBUTE_COLUMNS].max()
        merged_with_visual = merged.join(visual_seq, how="inner")

        print("=== 視覚属性の有無による平均検出率の差 (dateスプリットtest集合内) ===")
        print(f"(対象: {len(merged_with_visual)} シーケンス)\n")
        rows = []
        for col in VISUAL_ATTRIBUTE_COLUMNS:
            if col not in merged_with_visual.columns:
                continue
            with_attr = merged_with_visual.loc[merged_with_visual[col] == 1, "detection_rate"]
            without_attr = merged_with_visual.loc[merged_with_visual[col] == 0, "detection_rate"]
            rows.append({
                "attribute": col,
                "n_with": len(with_attr),
                "mean_detection_with": with_attr.mean(),
                "n_without": len(without_attr),
                "mean_detection_without": without_attr.mean(),
                "diff": (with_attr.mean() if len(with_attr) else float("nan"))
                - (without_attr.mean() if len(without_attr) else float("nan")),
            })
        attr_df = pd.DataFrame(rows).sort_values("diff")
        print(attr_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
    else:
        print(f"[skip] {visual_path} が無いため視覚属性による分析はスキップします\n")

    # 2. 天候別の平均検出率
    if "Weather" in merged.columns:
        print("=== 天候別の平均検出率 ===")
        print(
            merged.groupby("Weather")["detection_rate"]
            .agg(["mean", "count"])
            .sort_values("mean")
            .to_string(float_format=lambda v: f"{v:.3f}")
        )
        print()

    # 3. 会場別の平均検出率
    if "HillLocation" in merged.columns:
        print("=== 会場別の平均検出率 ===")
        print(
            merged.groupby("HillLocation")["detection_rate"]
            .agg(["mean", "count"])
            .sort_values("mean")
            .to_string(float_format=lambda v: f"{v:.3f}")
        )
        print()

    # 4. 検出率が低い順のシーケンス一覧(定性チェック対象の選定用)
    print("=== 検出率が低い順(下位10件) ===")
    cols_to_show = [
        c for c in ["AthleteName", "AthleteSurname", "HillLocation", "Weather", "detection_rate"]
        if c in merged.columns
    ]
    print(
        merged.sort_values("detection_rate")[cols_to_show]
        .head(10)
        .to_string(float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    )

    analysis_path = OUT_DIR / "detection_analysis.csv"
    merged.sort_values("detection_rate").to_csv(analysis_path, encoding="utf-8-sig")
    print(f"\n[saved] {analysis_path}")


if __name__ == "__main__":
    main()