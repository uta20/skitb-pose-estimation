"""
src.pipeline.process_sequence のテスト

実際のYOLOv8-pose推論はモデルロード・重みダウンロードのコストが大きいため、
ここでは「推定器を呼び出す前に早期リターンする分岐」だけを対象にする(アノテーションが無い/処理対象フレームが無いケース)
実際の推論を含むE2E動作確認は `scripts/run_pose_estimation.py` を手動実行して行う
"""
from pathlib import Path

from src.pipeline import process_sequence


def test_process_sequence_returns_error_when_sequence_missing(tmp_path: Path):
    result = process_sequence(
        sequence_id="JP9999",
        data_root=tmp_path,
        out_dir=tmp_path / "outputs",
        estimator=None,  # 早期リターンするため推定器は不要
    )

    assert result.sequence_id == "JP9999"
    assert result.frames_total == 0
    assert result.frames_detected == 0
    assert result.detection_rate == 0.0
    assert result.error is not None
    assert result.keypoints_csv is None

def test_process_sequence_returns_error_when_no_frame_images(tmp_path: Path):
    # MC/*.txt は用意するが、frames/ に画像を1枚も置かない
    mc_dir = tmp_path / "JP0001" / "MC"
    mc_dir.mkdir(parents=True)
    (mc_dir / "boxes.txt").write_text("0,0,10,10\n1,1,10,10\n", encoding="utf-8")
    (mc_dir / "frames.txt").write_text("1\n2\n", encoding="utf-8")
    (mc_dir / "visibilities.txt").write_text("1\n1\n", encoding="utf-8")
    (mc_dir / "cameras.txt").write_text("0\n0\n", encoding="utf-8")

    result = process_sequence(
        sequence_id="JP0001",
        data_root=tmp_path,
        out_dir=tmp_path / "outputs",
        estimator=None,
    )

    assert result.frames_total == 0
    assert "画像がありません" in result.error