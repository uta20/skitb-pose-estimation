from pathlib import Path

from src.metadata import (
    build_manifest,
    load_sequence_data,
    load_split,
    load_visual_attributes,
    split_membership,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def test_load_sequence_data_has_100_rows_and_known_athlete():
    df = load_sequence_data(DATA_DIR / "JP_data.csv")
    assert len(df) == 100
    assert df.loc["JP0001", "AthleteSurname"] == "Takanashi"
    # 末尾カンマ由来の無名列が残っていないこと
    assert not any(c.startswith("Unnamed") for c in df.columns)


def test_load_visual_attributes_splits_seq_and_camera_id():
    df = load_visual_attributes(DATA_DIR / "JP_visual_attributes.csv")
    row = df[(df["sequence_id"] == "JP0001") & (df["camera_id"] == 0)].iloc[0]
    assert row["SC"] == 1
    assert row["FOC"] == 0
    assert row["POC"] == 1
    # test集合(dateスプリット)の40シーケンス分のみ含まれる
    assert df["sequence_id"].nunique() == 40


def test_split_membership_lookup():
    split = load_split(
        DATA_DIR / "JP_train_test_date_60-40.json"
    )
    membership = split_membership(split)
    assert membership["JP0001"] == "test"
    assert membership["JP0060"] == "train"


def test_build_manifest_merges_all_six_splits():
    manifest = build_manifest(DATA_DIR, category="JP")
    assert len(manifest) == 100

    expected_cols = {
        "split_date_2way", "split_athlete_2way", "split_course_2way",
        "split_date_3way", "split_athlete_3way", "split_course_3way",
    }
    assert expected_cols.issubset(manifest.columns)

    # JP0001 は date-splitではtest、athlete/course-splitではtrainに属する
    # (アップロードされたJSON定義を参照)
    assert manifest.loc["JP0001", "split_date_2way"] == "test"
    assert manifest.loc["JP0001", "split_athlete_2way"] == "train"
    assert manifest.loc["JP0001", "split_course_2way"] == "train"
