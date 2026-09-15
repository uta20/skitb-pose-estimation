"""BBoxを用いた選手領域のクロップ処理。

姿勢推定モデル(YOLOv8-pose)における処理時間の短縮のために
SkiTBのアノテーションBBoxで選手領域を先に切り出す
ただし、BBoxの値で切り出すと手足の先端が
はみ出すことがあるため、一定のマージンを加える。
"""

from __future__ import annotations

import numpy as np

def expand_box(
    box_xyxy: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    """BBox (x1, y1, x2, y2) を上下左右に margin_ratio 分だけ拡張する。
    
        画像境界を超えないようクリップする。
        Parameters
        ----------
        box_xyxy: 元のBBox (x1, y1, x2, y2)
        image_size: (width, height)
        margin_ratio: BBoxの幅・高さに対するマージンの割合. 上下左右に25%ずつ拡張
    """
    x1, y1, x2, y2 = box_xyxy
    width, height = image_size
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin_ratio), int(h * margin_ratio)

    ex1 = max(0, x1 - mx)
    ey1 = max(0, y1 - my)
    ex2 = min(width, x2 + mx)
    ey2 = min(height, y2 + my)

    return ex1, ey1, ex2, ey2

def crop_image(image: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """NumPy配列の画像 (H, W, C) をBBoxで切り出す。"""
    x1, y1, x2, y2 = box_xyxy
    return image[y1:y2, x1:x2]