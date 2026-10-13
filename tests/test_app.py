import json
from io import BytesIO
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

import app as app_module
from app import _resolved_efficiency_loss, app
from dust_level import analyze_dust
from efficiency_model import SIMULATION_SOURCE, estimate_efficiency_loss
from gradcam import apply_dust_heatmap
from train import _metrics_from_predictions, split_flat_dataset
from video_processor import process_video, summarize_video_samples


def test_clean_predictions_zero_out_dust_coverage():
    analysis = analyze_dust(93.0, 20.8, 'Clean')

    assert analysis.dust_level == 'Clean'
    assert analysis.dust_coverage == 0.0


def test_gradcam_targets_the_predicted_class(monkeypatch):
    recorded = {}

    monkeypatch.setattr('gradcam.predict_tensor', lambda model, tensor: (0, 90.0, np.array([0.9, 0.1], dtype=np.float32)))
    monkeypatch.setattr('gradcam.detect_panel_mask', lambda image: np.ones((32, 32), dtype=np.uint8) * 255)
    monkeypatch.setattr('gradcam.read_bgr', lambda image_or_path: np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr('gradcam._find_target_layer', lambda model: object())

    class DummyGradCAM:
        def __init__(self, model, target_layer):
            pass

        def __call__(self, image_tensor, target_class=None):
            recorded['target_class'] = target_class
            return np.zeros((32, 32), dtype=np.float32)

        def remove_hooks(self):
            pass

    monkeypatch.setattr('gradcam.GradCAM', DummyGradCAM)

    apply_dust_heatmap(np.zeros((32, 32, 3), dtype=np.uint8), object())

    assert recorded['target_class'] == 0


def test_measured_efficiency_loss_uses_configured_readings(monkeypatch):
    monkeypatch.setenv('CLEAN_POWER_W', '1000')
    monkeypatch.setenv('DUSTY_POWER_W', '800')
    monkeypatch.setenv('CLEAN_IRRADIANCE_W_M2', '1000')
    monkeypatch.setenv('DUSTY_IRRADIANCE_W_M2', '1000')

    assert _resolved_efficiency_loss(0.0, 50.0) == 20.0


def test_random_forest_efficiency_estimate_uses_synthetic_coverage_rule():
    assert estimate_efficiency_loss(0.0) == 0.0
    assert estimate_efficiency_loss(100.0) == 20.0
    assert abs(estimate_efficiency_loss(50.0) - 10.0) <= 0.2
    assert SIMULATION_SOURCE.startswith("Synthetic Random Forest estimate")


def test_random_forest_efficiency_estimate_rejects_invalid_coverage():
    with pytest.raises(ValueError, match="between 0 and 100"):
        estimate_efficiency_loss(101.0)


def test_efficiency_loss_falls_back_to_labeled_estimate(monkeypatch):
    for name in (
        "CLEAN_POWER_W",
        "DUSTY_POWER_W",
        "CLEAN_IRRADIANCE_W_M2",
        "DUSTY_IRRADIANCE_W_M2",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _resolved_efficiency_loss(None, 50.0) == estimate_efficiency_loss(50.0)


def test_image_result_includes_synthetic_efficiency_estimate(monkeypatch, tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(app_module.cv2, 'imread', lambda *_args: image)
    monkeypatch.setattr(app_module.cv2, 'cvtColor', lambda value, _code: value)
    monkeypatch.setattr(app_module.cv2, 'imwrite', lambda *_args: True)
    monkeypatch.setattr(app_module, 'predict_rgb', lambda *_args: ('Dusty', 90.0, None))
    monkeypatch.setattr(
        app_module,
        'apply_dust_heatmap',
        lambda *_args: (image, image, 50.0),
    )
    monkeypatch.setattr(app_module, 'segmentation_model', None)
    monkeypatch.setattr(app_module, 'RESULT_DIR', tmp_path)
    monkeypatch.setattr(app_module, '_configured_efficiency_loss', lambda: None)

    result = app_module._image_result(Path('panel.jpg'), 'image-test')

    assert result['dust_coverage'] == 50.0
    assert abs(result['efficiency_loss'] - 10.0) <= 0.2
    assert result['efficiency_loss_source'] == SIMULATION_SOURCE


def test_video_worker_includes_synthetic_efficiency_estimate(
    monkeypatch, tmp_path
):
    output_path = tmp_path / 'processed.mp4'
    statuses = []
    monkeypatch.setattr(app_module, '_configured_efficiency_loss', lambda: None)
    monkeypatch.setattr(app_module, '_save_history', lambda *_args: None)
    monkeypatch.setattr(
        app_module,
        '_write_video_status',
        lambda _uid, payload: statuses.append(payload),
    )

    def summarize_video(**_kwargs):
        output_path.write_bytes(b'processed video')
        return {
            'overall_label': 'Dusty',
            'avg_coverage': 50.0,
            'avg_confidence': 90.0,
        }

    monkeypatch.setattr(app_module, 'process_video', summarize_video)
    app_module._run_video(
        tmp_path / 'input.mp4',
        output_path,
        'video-test',
        'admin',
        12,
        False,
    )

    result = statuses[-1]
    assert result['status'] == 'complete'
    assert result['dust_coverage'] == 50.0
    assert abs(result['efficiency_loss'] - 10.0) <= 0.2
    assert result['efficiency_loss_source'] == SIMULATION_SOURCE


def test_video_worker_keeps_clean_label_when_summary_is_clean(monkeypatch, tmp_path):
    output_path = tmp_path / 'processed.mp4'
    statuses = []
    monkeypatch.setattr(app_module, '_configured_efficiency_loss', lambda: None)
    monkeypatch.setattr(app_module, '_save_history', lambda *_args: None)
    monkeypatch.setattr(
        app_module,
        '_write_video_status',
        lambda _uid, payload: statuses.append(payload),
    )

    def summarize_video(**_kwargs):
        output_path.write_bytes(b'processed video')
        return {
            'overall_label': 'Clean',
            'avg_coverage': 34.8,
            'avg_confidence': 92.0,
        }

    monkeypatch.setattr(app_module, 'process_video', summarize_video)
    app_module._run_video(
        tmp_path / 'input.mp4',
        output_path,
        'clean-video-test',
        'admin',
        12,
        False,
    )

    result = statuses[-1]
    assert result['status'] == 'complete'
    assert result['label'] == 'Clean'
    assert result['dust_coverage'] == 0.0


def test_validation_metrics_match_confusion_matrix():
    metrics = _metrics_from_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        dataset_dir=Path('validation'),
        sample_count=4,
    )

    assert metrics['confusion_matrix'] == [[1, 1], [0, 2]]
    assert metrics['confusion_matrix_orientation'] == 'rows=actual, columns=predicted'
    assert metrics['accuracy'] == 75.0
    assert metrics['positive_class'] == 'Dusty'
    assert metrics['metric_average'] == 'binary'
    assert metrics['true_positive'] == 2
    assert metrics['false_negative'] == 0
    assert metrics['false_positive'] == 1
    assert metrics['true_negative'] == 1
    assert metrics['precision'] == 66.67
    assert metrics['recall'] == 100.0
    assert metrics['f1_score'] == 80.0
    assert metrics['specificity'] == 50.0
    assert metrics['macro_precision'] == 83.33
    assert metrics['macro_recall'] == 75.0
    assert metrics['macro_f1_score'] == 73.33
    assert metrics['per_class']['Clean'] == {
        'precision': 100.0,
        'recall': 50.0,
        'f1_score': 66.67,
        'support': 2,
    }
    assert metrics['per_class']['Dusty'] == {
        'precision': 66.67,
        'recall': 100.0,
        'f1_score': 80.0,
        'support': 2,
    }


def test_binary_dataset_split_ignores_other_class_folders(tmp_path):
    for class_name, color in [('Clean', 'white'), ('Dusty', 'gray'), ('Bird-drop', 'red')]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(5):
            Image.new('RGB', (8, 8), color=color).save(class_dir / f'{index}.png')

    train_set, test_set, classes = split_flat_dataset(tmp_path, test_fraction=0.2)

    assert classes == ('Clean', 'Dusty')
    assert len(train_set) == 8
    assert len(test_set) == 2
    assert set(train_set.dataset.classes) == {'Clean', 'Dusty'}
    assert set(test_set.dataset.classes) == {'Clean', 'Dusty'}


def test_evaluate_endpoint_returns_model_validation_metrics(monkeypatch):
    expected = {
        'accuracy': 75.0,
        'precision': 66.67,
        'recall': 100.0,
        'f1_score': 80.0,
        'true_positive': 2,
        'false_negative': 0,
        'false_positive': 1,
        'true_negative': 1,
        'confusion_matrix': [[1, 1], [0, 2]],
        'class_names': ['Clean', 'Dusty'],
        'sample_count': 4,
        'test_fraction': 0.2,
        'confusion_matrix_orientation': 'rows=actual, columns=predicted',
    }
    monkeypatch.setattr(app_module, 'evaluate_flat_dataset', lambda *args, **kwargs: expected)
    client = app.test_client()
    client.post('/login', data={'username': 'admin', 'password': 'admin123'})

    response = client.get('/evaluate')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['accuracy'] == expected['accuracy']
    assert payload['confusion_matrix'] == expected['confusion_matrix']
    assert payload['model'] == 'model.pth'
    assert payload['evaluation_type'] == 'held-out validation'


def test_video_reports_total_frame_count(tmp_path, monkeypatch):
    video_path = tmp_path / 'sample.mp4'
    output_path = tmp_path / 'processed.mp4'

    fourcc = 'mp4v'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (32, 32))
    for _ in range(8):
        frame = np.zeros((32, 32, 3), dtype='uint8')
        writer.write(frame)
    writer.release()

    monkeypatch.setattr('video_processor.predict_rgb', lambda model, frame_rgb: ('Clean', 93.0, None))

    summary = process_video(video_path, output_path, model=object(), skip_frames=0)

    assert summary['total_frames'] == 8
    assert summary['frames_processed'] >= 1
    assert summary['overall_label'] == 'Clean'
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert summary['output_video_codec'] == 'h264'
    assert summary['output_video_frames'] == summary['total_frames']

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        capture.release()

    reader = imageio_ffmpeg.read_frames(str(output_path))
    metadata = next(reader)
    first_frame = next(reader)
    reader.close()
    assert metadata['codec'] == 'h264'
    assert len(first_frame) > 0

    monkeypatch.setattr(app_module, 'VIDEO_RESULT_DIR', tmp_path)
    response = app.test_client().get(
        f'/videos/{output_path.name}',
        headers={'Range': 'bytes=0-31'},
    )
    assert response.status_code == 206
    assert response.mimetype == 'video/mp4'


def test_login_page_redirects_and_authenticates():
    client = app.test_client()

    resp = client.get('/')
    assert resp.status_code in (200, 302)

    login_resp = client.get('/login')
    assert login_resp.status_code == 200
    assert b'username' in login_resp.data.lower()

    auth_resp = client.post(
        '/login',
        data={'username': 'admin', 'password': 'admin123'},
        follow_redirects=True,
    )
    assert auth_resp.status_code == 200
    assert b'SolarSense Dashboard' in auth_resp.data


def test_registration_validates_creates_session_and_keeps_password_hashed(
    tmp_path, monkeypatch
):
    users_file = tmp_path / 'users.json'
    logins_file = tmp_path / 'login_history.json'
    monkeypatch.setattr(app_module, 'USERS_FILE', users_file)
    monkeypatch.setattr(app_module, 'LOGIN_HISTORY_FILE', logins_file)
    client = app.test_client()

    invalid = client.post(
        '/register',
        data={
            'username': 'solar-user',
            'password': 'long-enough-password',
            'confirm_password': 'different-password',
        },
    )

    assert invalid.status_code == 400
    assert b'Passwords do not match.' in invalid.data

    registered = client.post(
        '/register',
        data={
            'username': 'solar-user',
            'password': 'long-enough-password',
            'confirm_password': 'long-enough-password',
        },
    )

    assert registered.status_code == 302
    assert registered.headers['Location'].endswith('/dashboard')
    saved_users = json.loads(users_file.read_text())
    assert 'solar-user' in saved_users
    assert saved_users['solar-user']['password_hash'] != 'long-enough-password'
    assert client.get('/dashboard').status_code == 200
    assert client.get('/logout').status_code == 302


def test_history_endpoint_returns_list_for_logged_user():
    client = app.test_client()
    client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)

    resp = client.get('/history')
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_video_endpoint_rejects_non_mp4_uploads():
    client = app.test_client()
    client.post('/login', data={'username': 'admin', 'password': 'admin123'})

    response = client.post(
        '/predict_video',
        data={'video': (BytesIO(b'not an mp4'), 'panel.avi')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.get_json() == {'error': 'Unsupported video type.'}


def test_video_worker_reports_processing_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, 'VIDEO_RESULT_DIR', tmp_path)

    def fail_processing(**_kwargs):
        raise RuntimeError('sample processing failure')

    monkeypatch.setattr(app_module, 'process_video', fail_processing)
    output_path = tmp_path / 'result_failed.mp4'

    app_module._run_video(
        input_path=tmp_path / 'input.mp4',
        output_path=output_path,
        uid='failed',
        user='admin',
        skip=12,
        heatmap=False,
    )

    status = json.loads((tmp_path / 'failed_status.json').read_text())
    assert status['status'] == 'error'
    assert 'sample processing failure' in status['error']
    assert not output_path.exists()


def test_image_endpoint_accepts_jpg_upload(monkeypatch, tmp_path):
    client = app.test_client()
    client.post('/login', data={'username': 'admin', 'password': 'admin123'})
    monkeypatch.setattr(app_module, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(app_module, '_image_result', lambda path, uid: {'label': 'Clean'})
    monkeypatch.setattr(app_module, '_save_history', lambda user, result: None)

    response = client.post(
        '/predict',
        data={'file': (BytesIO(b'jpg image data'), 'panel.jpg')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    assert response.get_json() == {'label': 'Clean'}
    assert len(list(tmp_path.glob('*.jpg'))) == 1


def test_video_summary_reduces_false_positive_dusty_frames():
    frames = [
        {'label': 'Clean', 'confidence': 93.0, 'coverage': 1.2},
        {'label': 'Clean', 'confidence': 92.0, 'coverage': 2.1},
        {'label': 'Clean', 'confidence': 91.5, 'coverage': 1.4},
        {'label': 'Clean', 'confidence': 90.7, 'coverage': 2.6},
        {'label': 'Clean', 'confidence': 89.2, 'coverage': 3.2},
        {'label': 'Dusty', 'confidence': 68.2, 'coverage': 18.0},
        {'label': 'Clean', 'confidence': 90.0, 'coverage': 1.9},
        {'label': 'Clean', 'confidence': 91.5, 'coverage': 2.4},
        {'label': 'Clean', 'confidence': 89.6, 'coverage': 2.8},
        {'label': 'Clean', 'confidence': 90.8, 'coverage': 1.7},
        {'label': 'Clean', 'confidence': 90.1, 'coverage': 2.1},
        {'label': 'Clean', 'confidence': 92.3, 'coverage': 1.6},
    ]

    summary = summarize_video_samples(frames)

    assert summary['overall_label'] == 'Clean'
    assert summary['avg_coverage'] < 5.0
    assert summary['dusty_percent'] < 25.0
