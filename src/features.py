"""
COCO 17キーポイントから体の部位同士の相対角度(力学的特徴量)を計算

## 設計方針: なぜ「相対角度」に限定するか

SkiTBのカメラは選手を追って常にパン/チルトしており, 
画像の上方向と実世界の鉛直方向が一致しない
そのため, 絶対角度は物理量としての比較に使えない

本モジュールが計算する角度はすべて「体の部位同士の相対角度」であり, 
カメラの向きに依存しない特徴量である
また, 真の対気角度(angle of attack)を得るために必要な
カメラキャリブレーションによる3次元復元は本プロジェクトのスコープ外とする
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]
Keypoints = dict[str, tuple[float, float, float]]  # {name: (x, y, conf)}

DEFAULT_CONF_THRESHOLD = 0.3

@dataclass
class PoseFeatures:
    """1フレーム分の力学的特徴量"""

    trunk_hip_angle_deg: float | None       # 体幹の折れ角(肩中点-腰中点-膝中点)
    leg_v_angle_deg: float | None           # 脚のV字角(左脚ベクトルと右脚ベクトルのなす角)
    left_knee_angle_deg: float | None       # 左膝の伸展角(左腰-左膝-左足首)
    right_knee_angle_deg: float | None      # 右膝の伸展角(右腰-右膝-右足首)
    knee_asymmetry_deg: float | None        # |左膝角 - 右膝角|(左右非対称度)

def _get_point(
    keypoints: Keypoints,
    name: str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> Point | None:
    """信頼度が閾値以上なら (x, y) を返す"""
    if name not in keypoints:
        return None
    x, y, conf = keypoints[name]
    if conf < conf_threshold:
        return None
    return (x, y)

def _midpoint(p1: Point | None, p2: Point | None) -> Point | None:
    """中点計算"""
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def _angle_at_vertex(a: Point | None, vertex: Point | None, c: Point | None) -> float | None:
    """
    3点 a-vertex-c について, vertexにおける角度を返す

    いずれかの点が None(信頼度不足で未検出)の場合は None を返す
    3点が同一直線上に近い場合も None を返す
    """
    if a is None or vertex is None or c is None:
        return None

    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    return _angle_between(v1, v2)

def _angle_between(v1: Point, v2: Point) -> float | None:
    """2つのベクトルのなす角を返す. ベクトルの長さが0に近い場合はNone. """
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # 浮動小数点誤差での範囲外を防ぐ
    return math.degrees(math.acos(cos_theta))

def extract_features(
    keypoints: Keypoints,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> PoseFeatures:
    """1フレーム分のキーポイント辞書から力学的特徴量を計算

    Parameters
    ----------
    keypoints:
        ``{関節名: (x, y, confidence)}`` の辞書
        ``src.visualize.keypoints_to_dict()`` の出力形式 or 
        ``keypoints.csv`` の1行をパースしたものを想定
    conf_threshold:
        この信頼度未満のキーポイントは未検出として扱い, 
        それを使う特徴量は None になる
    """
    def pt(name: str) -> Point | None:
        return _get_point(keypoints, name, conf_threshold)

    left_shoulder, right_shoulder = pt("left_shoulder"), pt("right_shoulder")
    left_hip, right_hip = pt("left_hip"), pt("right_hip")
    left_knee, right_knee = pt("left_knee"), pt("right_knee")
    left_ankle, right_ankle = pt("left_ankle"), pt("right_ankle")

    mid_shoulder = _midpoint(left_shoulder, right_shoulder)
    mid_hip = _midpoint(left_hip, right_hip)
    mid_knee = _midpoint(left_knee, right_knee)

    # 体幹の折れ角: 肩中点-腰中点-膝中点. 180度に近いほど体が伸びていることを意味する
    trunk_hip_angle = _angle_at_vertex(mid_shoulder, mid_hip, mid_knee)

    # 脚のV字角: 腰から足首へのベクトルのなす角
    leg_v_angle = None
    if left_hip is not None and right_hip is not None:
        hip_mid = _midpoint(left_hip, right_hip)
        if hip_mid is not None and left_ankle is not None and right_ankle is not None:
            v_left = (left_ankle[0] - hip_mid[0], left_ankle[1] - hip_mid[1])
            v_right = (right_ankle[0] - hip_mid[0], right_ankle[1] - hip_mid[1])
            leg_v_angle = _angle_between(v_left, v_right)

    left_knee_angle = _angle_at_vertex(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle_at_vertex(right_hip, right_knee, right_ankle)

    knee_asymmetry = None
    if left_knee_angle is not None and right_knee_angle is not None:
        knee_asymmetry = abs(left_knee_angle - right_knee_angle)

    return PoseFeatures(
        trunk_hip_angle_deg=trunk_hip_angle,
        leg_v_angle_deg=leg_v_angle,
        left_knee_angle_deg=left_knee_angle,
        right_knee_angle_deg=right_knee_angle,
        knee_asymmetry_deg=knee_asymmetry,
    )