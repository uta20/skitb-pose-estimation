"""
src.features のテスト

実際のYOLOv8-poseの出力ではなく, 角度があらかじめ分かっている
合成座標(直角・直線などのキリの良い配置)を使い, 
計算結果が幾何学的に正しいことを検証する
"""

from src.features import extract_features


def _kp(x: float, y: float, conf: float = 0.9) -> tuple[float, float, float]:
    return (x, y, conf)

def _assert_close(actual: float | None, expected: float, tol: float = 1e-6) -> None:
    assert actual is not None
    assert abs(actual - expected) < tol

def test_trunk_hip_angle_is_180_when_body_is_straight():
    # 肩中点-腰中点-膝中点が一直線 -> 180度
    keypoints = {
        "left_shoulder": _kp(-5, 0), "right_shoulder": _kp(5, 0),
        "left_hip": _kp(-5, 10), "right_hip": _kp(5, 10),
        "left_knee": _kp(-5, 20), "right_knee": _kp(5, 20),
    }
    features = extract_features(keypoints)
    _assert_close(features.trunk_hip_angle_deg, 180.0)

def test_trunk_hip_angle_is_90_when_bent_forward_at_right_angle():
    # 腰(0,10)を頂点に, 肩が真横(-10,10), 膝が真下(0,20) -> 直角に前傾
    keypoints = {
        "left_shoulder": _kp(-10, 10), "right_shoulder": _kp(-10, 10),
        "left_hip": _kp(0, 10), "right_hip": _kp(0, 10),
        "left_knee": _kp(0, 20), "right_knee": _kp(0, 20),
    }
    features = extract_features(keypoints)
    _assert_close(features.trunk_hip_angle_deg, 90.0)

def test_leg_v_angle_is_90_for_symmetric_v_shape():
    # 腰(0,0)から見て左右対称に開いた脚 -> V字角90度
    keypoints = {
        "left_hip": _kp(-2, 0), "right_hip": _kp(2, 0),
        "left_ankle": _kp(-10, 10), "right_ankle": _kp(10, 10),
    }
    features = extract_features(keypoints)
    _assert_close(features.leg_v_angle_deg, 90.0)

def test_leg_v_angle_is_0_when_legs_are_together():
    # 左右の足首が同じ位置(脚を閉じている) -> V字角0度
    keypoints = {
        "left_hip": _kp(-2, 0), "right_hip": _kp(2, 0),
        "left_ankle": _kp(0, 10), "right_ankle": _kp(0, 10),
    }
    features = extract_features(keypoints)
    _assert_close(features.leg_v_angle_deg, 0.0)

def test_knee_angle_is_180_for_straight_leg_and_90_for_bent_leg():
    keypoints = {
        # 左脚: 腰-膝-足首が一直線 -> 180度
        "left_hip": _kp(0, 0), "left_knee": _kp(0, 10), "left_ankle": _kp(0, 20),
        # 右脚: 膝(0,10)を頂点に, 腰(0,0)の真下, 足首が真横(10,10) -> 直角に曲がっている
        "right_hip": _kp(0, 0), "right_knee": _kp(0, 10), "right_ankle": _kp(10, 10),
    }
    features = extract_features(keypoints)
    _assert_close(features.left_knee_angle_deg, 180.0)
    _assert_close(features.right_knee_angle_deg, 90.0)
    _assert_close(features.knee_asymmetry_deg, 90.0)

def test_low_confidence_keypoint_is_treated_as_missing():
    keypoints = {
        "left_shoulder": _kp(-5, 0, conf=0.05),  # 信頼度が閾値未満 -> 未検出扱い
        "right_shoulder": _kp(5, 0, conf=0.9),
        "left_hip": _kp(-5, 10, conf=0.9), "right_hip": _kp(5, 10, conf=0.9),
        "left_knee": _kp(-5, 20, conf=0.9), "right_knee": _kp(5, 20, conf=0.9),
    }
    features = extract_features(keypoints)
    # mid_shoulderが計算できないため, trunk_hip_angleもNoneになる
    assert features.trunk_hip_angle_deg is None

def test_missing_keypoints_return_none_without_raising():
    features = extract_features({})
    assert features.trunk_hip_angle_deg is None
    assert features.leg_v_angle_deg is None
    assert features.left_knee_angle_deg is None
    assert features.right_knee_angle_deg is None
    assert features.knee_asymmetry_deg is None