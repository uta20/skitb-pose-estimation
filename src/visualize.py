"""キーポイントとBBoxをフレーム画像上に描画するユーティリティ"""

from __future__ import annotations
import cv2
import numpy as np
from .pose import COCO_KEYPOINT_NAMES, PoseResult

# COCO 17キーポイントの骨格接続(index同士)
SKELETON = [
    (5, 7), (7, 9),        # 左肩-左肘-左手首
    (6, 8), (8, 10),       # 右肩-右肘-右手首
    (5, 6),                # 両肩
    (5, 11), (6, 12),      # 肩-腰
    (11, 12),              # 両腰
    (11, 13), (13, 15),    # 左腰-左膝-左足首
    (12, 14), (14, 16),    # 右腰-右膝-右足首
    (0, 5), (0, 6),        # 鼻-両肩
]

KEYPOINT_COLOR = (0, 220, 255)   # BGR: 黄色系
SKELETON_COLOR = (60, 200, 60)   # BGR: 緑
BOX_COLOR = (255, 80, 80)        # BGR: 青系


def draw_pose(
    image_bgr: np.ndarray,
    pose: PoseResult,
    box_xyxy: tuple[int, int, int, int] | None = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """画像をコピーしてキーポイント, 骨格, BBoxを描画したものを返す"""
    vis = image_bgr.copy()

    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2)

    pts = pose.keypoints_xy
    confs = pose.keypoints_conf

    for i, j in SKELETON:
        if confs[i] < conf_threshold or confs[j] < conf_threshold:
            continue
        p1 = tuple(pts[i].astype(int))
        p2 = tuple(pts[j].astype(int))
        cv2.line(vis, p1, p2, SKELETON_COLOR, 2, lineType=cv2.LINE_AA)

    for idx, (x, y) in enumerate(pts):
        if confs[idx] < conf_threshold:
            continue
        cv2.circle(vis, (int(x), int(y)), 4, KEYPOINT_COLOR, -1, lineType=cv2.LINE_AA)

    return vis


def keypoints_to_dict(pose: PoseResult) -> dict[str, tuple[float, float, float]]:
    """{関節名: (x, y, confidence)} の辞書に変換する"""
    return {
        name: (float(x), float(y), float(c))
        for name, (x, y), c in zip(
            COCO_KEYPOINT_NAMES, pose.keypoints_xy, pose.keypoints_conf
        )
    }