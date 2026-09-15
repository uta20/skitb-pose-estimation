"""SkiTBのシーケンスフォルダを管理するための軽量ローダー

SkiTB の各シーケンスフォルダは以下の構造を持つ:
    JP00xx/
      frames/            *.jpg フレーム画像
      MC/                MC(マルチカメラ)設定でのアノテーション
        boxes.txt        [x, y, w, h] 各行1フレーム分のBBox(左上座標+幅高さ)
        frames.txt       元動画に対するフレーム番号
        visibilities.txt 各フレームの可視性(1=50%以上可視 / 0=それ以外)
        cameras.txt      各フレームを撮影したカメラのインデックス(Top=0)
      SC/<camera_id>/    SC(単一カメラ)設定での同様のアノテーション

本モジュールはこれらのテキストファイルを読み込み、フレーム番号をキーに
BBox・可視性・カメラIDへ簡単にアクセスできる `SkiSequence` を提供する。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class FrameAnnotation:
    """1フレーム分のアノテーション"""

    frame_id: int          # フレーム番号 (frames.txt)
    box: tuple[int, int, int, int]  # (x, y, w, h)
    visible: bool           # visibilities.txt (1 -> True)
    camera_id: int | None   # cameras.txt (MC設定のみ存在)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) 形式のBBoxを返す。"""
        x, y, w, h = self.box
        return x, y, x + w, y + h

class SkiSequence:
    """1つのMC (またはSC) シーケンスフォルダを表すクラス。

    Examples
    --------
    >>> seq = SkiSequence.from_mc_dir(Path("data/JP0001"))
    >>> ann = seq.get(63)
    >>> ann.box
    (590, 168, 191, 404)
    """

    def __init__(
        self,
        sequence_id: str,
        annotations: list[FrameAnnotation],
        frames_dir: Path,
    ) -> None:
        self.sequence_id = sequence_id
        self.frames_dir = frames_dir
        self._by_frame_id = {a.frame_id: a for a in annotations}

    def __len__(self) -> int:
        return len(self._by_frame_id)

    def __iter__(self):
        return iter(sorted(self._by_frame_id.values(), key=lambda a: a.frame_id))

    def get(self, frame_id: int) -> FrameAnnotation:
        return self._by_frame_id[frame_id]

    def frame_path(self, frame_id: int) -> Path:
        """
        frame_id に対応する画像ファイルパスを返す
        SkiTB の frames/ ディレクトリは `%05d.jpg` 形式のファイル名を持つ
        """
        return self.frames_dir / f"{frame_id:05d}.jpg"

    def available_frame_ids(self) -> list[int]:
        """
        実際に画像ファイルが存在するフレームIDのみを返す
        本リポジトリのサンプルデータには全フレーム画像は含まれておらず、
        デモ用に1枚 (00063.jpg) のみを同梱しているため、このヘルパーで
        今すぐ処理できるフレームを絞り込む
        """
        return [
            fid for fid in self._by_frame_id
            if self.frame_path(fid).exists()
        ]

    @classmethod
    def from_mc_dir(cls, sequence_dir: Path) -> SkiSequence:
        """`<sequence_dir>/MC/*.txt` からシーケンスを構築する"""
        mc_dir = sequence_dir / "MC"
        frames_dir = sequence_dir / "frames"

        frame_ids = _read_int_lines(mc_dir / "frames.txt")
        boxes = _read_box_lines(mc_dir / "boxes.txt")
        visibilities = _read_int_lines(mc_dir / "visibilities.txt")
        cameras = _read_int_lines(mc_dir / "cameras.txt")

        _assert_same_length(frame_ids, boxes, visibilities, cameras)

        annotations = [
            FrameAnnotation(
                frame_id=fid,
                box=box,
                visible=bool(vis),
                camera_id=cam,
            )
            for fid, box, vis, cam in zip(frame_ids, boxes, visibilities, cameras)
        ]
        return cls(sequence_dir.name, annotations, frames_dir)

    @classmethod
    def from_sc_dir(cls, sequence_dir: Path, camera_id: int) -> SkiSequence:
        """
        `<sequence_dir>/SC/<camera_id>/*.txt` からシーケンスを構築する

        SC設定には cameras.txt が存在しないため、指定された camera_id を
        全フレームに割り当てる
        """
        sc_dir = sequence_dir / "SC" / str(camera_id)
        frames_dir = sequence_dir / "frames"

        frame_ids = _read_int_lines(sc_dir / "frames.txt")
        boxes = _read_box_lines(sc_dir / "boxes.txt")
        visibilities = _read_int_lines(sc_dir / "visibilities.txt")

        _assert_same_length(frame_ids, boxes, visibilities)

        annotations = [
            FrameAnnotation(
                frame_id=fid,
                box=box,
                visible=bool(vis),
                camera_id=camera_id,
            )
            for fid, box, vis in zip(frame_ids, boxes, visibilities)
        ]
        return cls(f"{sequence_dir.name}_cam{camera_id}", annotations, frames_dir)
    

def _read_int_lines(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _read_box_lines(path: Path) -> list[tuple[int, int, int, int]]:
    boxes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, w, h = (int(v) for v in line.split(","))
            boxes.append((x, y, w, h))
    return boxes

def _assert_same_length(*lists: list) -> None:
    lengths = {len(lst) for lst in lists}
    if len(lengths) != 1:
        raise ValueError(
            f"アノテーションファイル間で行数が一致しません: {lengths}"
        )