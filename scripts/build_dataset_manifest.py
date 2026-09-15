"""
JP_data.csv, 6種類のsplit定義JSON, JP_visual_attributes.csv を統合し, 
姿勢推定パイプラインで最初に処理すべきシーケンスの特定材料となる
マニフェストCSVを生成するスクリプト

Inputs: 
    - data/metadata/JP_data.csv
    - data/metadata/splits_JP_*.json (6種類)
    - data/metadata/JP_visual_attributes.csv

処理内容:
1. メタデータと分割情報の統合
   ``JP_data.csv`` を読み込み, 6種類の分割列(date/athlete/course x 2way/3way)を
   シーケンス単位に結合(全100行)

2. カメラ単位視覚属性のシーケンス集約
   ``JP_visual_attributes.csv`` (date splitのtest集合)を読み込み, 論理和でシーケンス単位に集約
   (例: いずれかのカメラでFOC=1なら「そのシーケンスは完全遮蔽を含む」)

3. データ結合と保存
   1のメタデータ と 2の視覚属性を ``sequence_id`` で結合し, ``outputs/manifest.csv`` に保存

4. ベースライン評価用のシーケンス抽出
   姿勢推定の「次に処理すべきシーケンス」の一例として, 
   date-splitのtest集合のうち FOC, POC, MBが0 のシーケンスを抽出して表示する

outputs: 
    - outputs/manifest.csv: 統合されたマニフェストCSV

使い方:
    uv run scripts/build_dataset_manifest.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.metadata import build_manifest, load_visual_attributes

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" 
OUT_DIR = REPO_ROOT / "outputs"

def summarize_visual_attributes(visual_attrs: pd.DataFrame) -> pd.DataFrame:
    """
    カメラ単位の視覚属性をシーケンス単位に集約

    いずれかのカメラのクリップでその属性が真であれば, 
    シーケンス全体としてその困難さを含む
    とみなし, 論理和(OR)で集約する
    """
    attribute_cols = [
        c for c in visual_attrs.columns
        if c not in ("sequence_id", "camera_id")
    ]
    agg = (
        visual_attrs.groupby("sequence_id")[attribute_cols]
        .max()  # attribute_colsの論理和
        .add_prefix("any_")
    )
    agg["num_sc_clips"] = visual_attrs.groupby("sequence_id").size() #グループサイズを記録
    return agg

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. シーケンス単位メタデータ + 6種split
    manifest = build_manifest(DATA_DIR, category="JP")
    print(f"[manifest] {len(manifest)} シーケンス, {manifest.shape[1]} 列")

    # 2. 視覚属性 (date-splitのtest集合のみ存在)の集約
    visual_attrs = load_visual_attributes(DATA_DIR / "JP_visual_attributes.csv")
    visual_summary = summarize_visual_attributes(visual_attrs)
    print(
        f"[visual_attributes] {visual_attrs['sequence_id'].nunique()} シーケンス "
        f"(dateスプリットのtest集合), 計{len(visual_attrs)} SCクリップ"
    )

    # 3. 結合(視覚属性が無いシーケンスはNaNのまま残す = train集合など)と保存
    manifest = manifest.join(visual_summary, how="left") # sequence_idで左側結合

    manifest_path = OUT_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, encoding="utf-8-sig")
    print(f"[saved] {manifest_path}")

    # 4. ベースライン候補の抽出
    #    date-split の test 集合で完全遮蔽(FOC), 部分遮蔽(POC), モーションブラー(MB)が0のシーケンス 
    # 　　= 姿勢推定のベースライン評価に適したシーケンス
    is_test_date = manifest["split_date_2way"] == "test"
    has_visual_attrs = manifest["num_sc_clips"].notna()
    is_easy = (
        (manifest.get("any_FOC", 0) == 0)
        & (manifest.get("any_POC", 0) == 0)
        & (manifest.get("any_MB", 0) == 0)
    )

    candidates = manifest.loc[is_test_date & has_visual_attrs & is_easy]
    print(
        f"\n[候補] date-split test集合 かつ 完全遮蔽/部分遮蔽/ブレなし: "
        f"{len(candidates)} シーケンス"
    )
    cols_to_show = ["AthleteName", "AthleteSurname", "HillLocation", "Weather"]
    cols_to_show = [c for c in cols_to_show if c in candidates.columns]
    print(candidates[cols_to_show].to_string())


if __name__ == "__main__":
    main()