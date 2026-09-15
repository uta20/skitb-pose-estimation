"""Ultralytics YOLOv8-pose を用いた姿勢推定のクラス群を提供するモジュール

COCO形式の17キーポイント (鼻, 両目, 両耳, 両肩, 両肘, 両手首, 両腰, 両膝, 両足首)を推定する
モデル自体は日常動作の人物画像で学習されているため, 
スキージャンプの空中姿勢(V字開脚・前傾姿勢)のような極端な姿勢に対しては精度が低下することが予想される
本プロジェクトではCOCO学習済みモデルをベースラインとして導入し, 
精度評価の結果次第でドメイン特化のファインチューニングを行う方針とする(README参照)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# COCO 17キーポイントの名称(Ultralytics/YOLOv8-poseの出力順)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

@dataclass
class PoseResult:
    """
    1人分の姿勢推定結果
    座標はクロップ前の元画像座標系に変換済み
    """

    keypoints_xy: np.ndarray   
    keypoints_conf: np.ndarray  
    bbox_conf: float

class PoseEstimator:
    """YOLOv8-pose のロードと推論をまとめたラッパークラス"""

    def __init__(self, model_name: str = "yolov8n-pose.pt", device: str = "cpu") -> None:
        # ultralytics/torchの読み込みコストを避けるために, インスタンス化時に遅延importする
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device

    def predict(
        self,
        crop_bgr: np.ndarray,
        offset_xy: tuple[int, int] = (0, 0),
        conf: float = 0.25,
    ) -> PoseResult | None:
        """
        クロップ画像に対して姿勢推定を行い, 元画像の座標系に変換した結果を返す

        Parameters
        ----------
        crop_bgr: OpenCVで読み込んだBGR画像(選手領域を切り出したもの)
        offset_xy: クロップ画像の元画像内での左上座標 (x, y)
        conf: 検出の信頼度しきい値
        """
        results = self.model.predict(
            crop_bgr, device=self.device, conf=conf, verbose=False
        )
        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # 複数人検出された場合, BBox信頼度が最大の人物を採用
        boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
        if boxes_conf is not None and len(boxes_conf) > 0:
            best_idx = int(np.argmax(boxes_conf))
            best_conf = float(boxes_conf[best_idx])
        else:
            best_idx = 0
            best_conf = 0.0

        kp_xy = result.keypoints.xy.cpu().numpy()[best_idx]      
        kp_conf = result.keypoints.conf.cpu().numpy()[best_idx]  

        ox, oy = offset_xy
        kp_xy_original = kp_xy + np.array([ox, oy])

        return PoseResult(
            keypoints_xy=kp_xy_original,
            keypoints_conf=kp_conf,
            bbox_conf=best_conf,
        )