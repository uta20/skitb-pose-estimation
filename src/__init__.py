"""
SkiTBデータセットを用いたスキージャンプ選手の姿勢推定パイプライン

サブモジュールが提供する主要なクラス・関数をまとめてexport

- dataset:   SkiTBアノテーション(MC/SC)のパーサ、シーケンス自動検出
- metadata:  シーケンス単位メタデータ・train/test分割・視覚属性の統合
- crop:      BBoxクロップ・マージン付与
- pose:      YOLOv8-poseラッパー
- pipeline:  1シーケンス分の姿勢推定処理(単一/バッチ両対応)
- visualize: キーポイント・骨格の描画
"""

from .dataset import FrameAnnotation, SkiSequence
from .metadata import (
    build_manifest,
    load_sequence_data,
    load_split,
    load_visual_attributes,
    split_membership,
)
from .pose import PoseEstimator, PoseResult
from .visualize import draw_pose, keypoints_to_dict

__all__ = [
    "FrameAnnotation",
    "PoseEstimator",
    "PoseResult",
    "SkiSequence",
    "build_manifest",
    "draw_pose",
    "keypoints_to_dict",
    "load_sequence_data",
    "load_split",
    "load_visual_attributes",
    "split_membership",
]