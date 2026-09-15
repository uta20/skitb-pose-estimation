"""SkiTBのシーケンス単位メタデータ・train/test分割定義を扱うモジュール。

対象ファイル:

- ``JP_data.csv``
    各シーケンスの属性(選手名・国籍・大会会場・天候・採点・結果など)。
    1行 = 1シーケンス(例: JP0001)。

- ``JP_visual_attributes.csv``
    ``JP_train_test_date_60-40.json`` の **test集合に含まれるシーケンス**
    についてのみ、SCクリップ(カメラ毎に分割したクリップ)単位で
    付与された視覚的困難度の属性(VOTベンチマーク系の属性セットに準拠):

    ============  =====================================================
    列名           意味
    ============  =====================================================
    SC             Scale Change (スケール変化)
    ARC            Aspect Ratio Change (アスペクト比変化)
    LR             Low Resolution (低解像度)
    FM             Fast Motion (高速な動き)
    CM             Camera Motion (カメラの動き)
    FOC            Full Occlusion (完全遮蔽)
    POC            Partial Occlusion (部分遮蔽)
    IV             Illumination Variation (照明変化)
    BC             Background Clutter (背景の複雑さ)
    MB             Motion Blur (モーションブラー)
    ============  =====================================================

    行キー ``seq_names`` は ``<sequence_id>_<camera_id>`` 形式
    (例: ``JP0001_0`` は JP0001 のカメラ0によるSCクリップ)。

- ``JP_train_test_<criterion>_60-40.json`` /
  ``JP_train_val_test_<criterion>_60-40.json``
    ``criterion`` は ``date`` (大会日)・``athlete`` (選手)・``course``
    (会場) のいずれかで、train/(val)/test にシーケンスIDを振り分けた
    定義ファイル。
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# JP_visual_attributes.csv の属性列とその意味(README/参照コメント用)
VISUAL_ATTRIBUTE_COLUMNS = [
    "SC", "ARC", "LR", "FM", "CM", "FOC", "POC", "IV", "BC", "MB",
]

def load_sequence_data(csv_path: Path) -> pd.DataFrame:
    """
    ``JP_data.csv`` を読み込み、``ID`` (例: JP0001) を
    インデックスとした DataFrame を返す

    csvの末尾に付与されている空列は破棄する
    """
    df = pd.read_csv(csv_path)
    # 末尾の "Unnamed: N" 空列を除去
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.set_index("ID")

    # 日付分析のためにDate列 (例: 20220101) を datetime に変換しておく
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")

    return df

def load_visual_attributes(csv_path: Path) -> pd.DataFrame:
    """
    ``JP_visual_attributes.csv`` を読み込む

    このCSVは他ファイルと異なるフォーマットのため、専用の読み込み処理を用意
    ``seq_names`` (例: ``JP0001_0``) を
    ``sequence_id`` (``JP0001``) と ``camera_id`` (``0``) に分割し、
    属性列は 0/1 の真偽値として扱えるよう整数型に変換する
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    split_names = df["seq_names"].str.rsplit("_", n=1, expand=True)
    df.insert(0, "sequence_id", split_names[0])
    df.insert(1, "camera_id", split_names[1].astype(int))
    df = df.drop(columns=["seq_names"])

    for col in VISUAL_ATTRIBUTE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)

    return df

def load_split(json_path: Path) -> dict[str, list[str]]:
    """
    train/(val)/test の分割定義JSONを読み込む

    2分割 (``train``/``test``) と3分割 (``train``/``val``/``test``) の
    両方の形式をそのまま辞書として返す
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)

def split_membership(split: dict[str, list[str]]) -> dict[str, str]:
    """
    ``{sequence_id: サブセット名}`` の逆引き辞書を作成

    例: ``{"JP0001": "test", "JP0002": "train", ...}``
    """
    membership: dict[str, str] = {}
    for subset_name, sequence_ids in split.items():
        for seq_id in sequence_ids:
            membership[seq_id] = subset_name
    return membership

def build_manifest(
    data_dir: Path,
    category: str = "JP",
) -> pd.DataFrame:
    """メタデータ一式を1つのDataFrameに統合した「マニフェスト」を作成

    ``<sequence_data>`` に加えて、6種類のsplit定義それぞれについて
    そのシーケンスが train/val/test のどれに属するかを列として付与する
    列名は ``split_<criterion>_<2way|3way>`` の形式
    (例: ``split_date_2way``, ``split_athlete_3way``)

    可視属性(visual_attributes)は test 集合のシーケンスにしか存在せず、
    かつ1シーケンスにつき複数カメラ分の行を持つため、この統合
    マニフェストには含めない. 
    個別に ``load_visual_attributes`` で取得し、
    ``sequence_id`` で結合(merge)して利用すること
    """
    data_dir = Path(data_dir)
    manifest = load_sequence_data(data_dir / f"{category}_data.csv")

    split_specs = [
        ("date", "2way", f"{category}_train_test_date_60-40.json"),
        ("athlete", "2way", f"{category}_train_test_athlete_60-40.json"),
        ("course", "2way", f"{category}_train_test_course_60-40.json"),
        ("date", "3way", f"{category}_train_val_test_date_60-40.json"),
        ("athlete", "3way", f"{category}_train_val_test_athlete_60-40.json"),
        ("course", "3way", f"{category}_train_val_test_course_60-40.json"),
    ]

    for criterion, arity, filename in split_specs:
        split = load_split(data_dir / filename)
        membership = split_membership(split)
        col = f"split_{criterion}_{arity}"
        manifest[col] = manifest.index.map(membership)

    return manifest