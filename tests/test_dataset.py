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


def test_available_frame_ids_includes_first_frame_of_jp0001():
    """JP0001の最初のフレーム(frame 63)が必ず含まれることを確認する

    ローカル環境によって同梱されるフレーム数が異なりうるため, 
    件数によらず常に成り立つべき性質だけを検証する. 
    """
    seq = SkiSequence.from_mc_dir(REPO_ROOT / "data" / "JP0001")
    available = seq.available_frame_ids()

    # frame 63 は常に含まれるはず
    assert 63 in available

    # 返された各frame_idについて、実際に画像ファイルが存在すること
    for frame_id in available:
        assert seq.frame_path(frame_id).exists()

    # 返されたframe_idはすべて、アノテーション(frames.txt)に含まれること
    annotated_ids = {a.frame_id for a in seq}
    assert set(available).issubset(annotated_ids)

    # フルデータ(317フレーム)が揃っている場合は、317件すべて検出できること
    if len(available) > 1:
        assert len(available) == 317
        assert available == sorted(available)


def test_expand_box_clips_to_image_bounds():
    # 画像左上ギリギリのBBoxを大きく拡張しても、負の座標にならないこと
    box = (5, 5, 50, 100)
    expanded = expand_box(box, image_size=(1280, 720), margin_ratio=0.5)
    x1, y1, x2, y2 = expanded
    assert x1 == 0
    assert y1 == 0
    assert x2 <= 1280
    assert y2 <= 720
