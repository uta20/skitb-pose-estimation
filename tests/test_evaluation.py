import pandas as pd
import pytest

from src.evaluation import (
    ConfusionCounts,
    compute_confusion,
    keypoint_confidence_stats,
)


def test_confusion_counts_properties():
    c = ConfusionCounts(tp=8, fn=2, fp=1, tn=9)
    assert c.total == 20
    assert c.recall == pytest.approx(8 / 10)
    assert c.precision == pytest.approx(8 / 9)
    assert c.accuracy == pytest.approx(17 / 20)


def test_confusion_counts_handles_zero_denominator():
    # 可視フレームが1件も無い(tp+fn=0)場合、recallは定義できずNone
    c = ConfusionCounts(tp=0, fn=0, fp=0, tn=5)
    assert c.recall is None
    assert c.precision is None
    assert c.accuracy == pytest.approx(1.0)


def test_confusion_counts_addition():
    a = ConfusionCounts(tp=1, fn=0, fp=0, tn=0)
    b = ConfusionCounts(tp=2, fn=1, fp=0, tn=0)
    total = a + b
    assert total.tp == 3
    assert total.fn == 1
    assert total.total == 4


def test_compute_confusion_basic():
    visibility = {1: True, 2: True, 3: False, 4: False}
    detected = {1: True, 2: False, 3: True, 4: False}

    counts = compute_confusion(visibility, detected)

    assert counts.tp == 1  # frame 1: 可視かつ検出
    assert counts.fn == 1  # frame 2: 可視だが見逃し
    assert counts.fp == 1  # frame 3: 遮蔽だが検出
    assert counts.tn == 1  # frame 4: 遮蔽かつ未検出


def test_compute_confusion_ignores_frames_missing_from_either_side():
    visibility = {1: True, 2: True}
    detected = {1: True, 3: True}  # frame 2, 3 は片方にしか存在しない

    counts = compute_confusion(visibility, detected)

    assert counts.total == 1  # 共通するframe 1のみが対象
    assert counts.tp == 1


def test_keypoint_confidence_stats_only_uses_detected_frames():
    df = pd.DataFrame({
        "frame_id": [1, 2, 3],
        "detected": [True, True, False],
        "nose_x": [10, 20, None],
        "nose_y": [10, 20, None],
        "nose_conf": [0.9, 0.5, None],
        "left_eye_x": [11, 21, None],
        "left_eye_y": [11, 21, None],
        "left_eye_conf": [0.8, 0.4, None],
    })

    stats = keypoint_confidence_stats(df)

    nose_row = stats[stats["keypoint"] == "nose"].iloc[0]
    # frame 3 は detected=False のため集計対象から除外される
    assert nose_row["n_frames"] == 2
    assert nose_row["mean_conf"] == pytest.approx(0.7)


def test_keypoint_confidence_stats_sorted_by_mean_conf_ascending():
    df = pd.DataFrame({
        "frame_id": [1],
        "detected": [True],
        "nose_x": [10], "nose_y": [10], "nose_conf": [0.9],
        "left_ankle_x": [10], "left_ankle_y": [10], "left_ankle_conf": [0.3],
    })

    stats = keypoint_confidence_stats(df)

    # 信頼度が低い関節(left_ankle)が先頭に来ること
    assert stats.iloc[0]["keypoint"] == "left_ankle"
    assert stats.iloc[-1]["keypoint"] == "nose"