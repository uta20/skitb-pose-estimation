from pathlib import Path

from src.crop import expand_box
from src.dataset import SkiSequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_jp0001_mc_sequence():
    seq = SkiSequence.from_mc_dir(REPO_ROOT / "data" / "JP0001")
    # frames.txt: 63〜379 の317フレーム
    assert len(seq) == 317

    first = seq.get(63)
    assert first.box == (590, 168, 191, 404)
    assert first.visible is True
    assert first.camera_id == 0


def test_available_frame_ids_returns_first_frame_of_jp0001():
    """
    JP0001の最初のフレーム(frame 63, 00063.jpg)のみを対象にしたテスト

    現時点ではデモ用に frame 63 の画像のみを同梱
    JP0001の全317フレーム, および将来的なJP0002〜JP0100を用いた
    フルデータでの動作確認は別途行う
    """
    seq = SkiSequence.from_mc_dir(REPO_ROOT / "data" / "JP0001")

    available = seq.available_frame_ids()
    assert available == [63]

    first = seq.get(available[0])
    assert first.frame_id == 63
    assert seq.frame_path(63).exists()


def test_expand_box_clips_to_image_bounds():
    # 画像左上ギリギリのBBoxを大きく拡張しても、負の座標にならないこと
    box = (5, 5, 50, 100)
    expanded = expand_box(box, image_size=(1280, 720), margin_ratio=0.5)
    x1, y1, x2, y2 = expanded
    assert x1 == 0
    assert y1 == 0
    assert x2 <= 1280
    assert y2 <= 720
