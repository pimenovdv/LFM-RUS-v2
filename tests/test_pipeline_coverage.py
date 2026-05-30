import json
from src.data_prep.pipeline import run_data_prep_pipeline

def test_run_data_prep_pipeline(tmp_path):
    # Create dummy raw data
    input_dir = tmp_path / "data" / "raw"
    input_dir.mkdir(parents=True)
    raw_file = input_dir / "data.jsonl"
    with open(raw_file, "w") as f:
        f.write(json.dumps({"text": "Hello world", "id": "1"}) + "\n")
        f.write(json.dumps({"text": "Hello python", "id": "2"}) + "\n")
        f.write(json.dumps({"text": "Hello python", "id": "3"}) + "\n") # Duplicate

    output_dir = tmp_path / "data" / "deduplicated"
    minhash_dir = tmp_path / "data" / "minhash"

    cfg = {
        "input_path": str(input_dir),
        "output_path": str(output_dir),
        "minhash_base_path": str(minhash_dir),
        "minhash_config": {
            "n_grams": 2,
            "num_buckets": 2,
            "hashes_per_bucket": 2,
            "precision": 64
        },
        "filters": {
            "remove_seo": True,
            "remove_logs": True,
            "remove_cyclic": True,
            "fasttext_spam": {
                "enabled": False # Disable since it might require downloading real model
            },
            "fineweb_quality": {
                "enabled": True
            }
        }
    }

    run_data_prep_pipeline(cfg)

    # Check that output file exists
    final_output_file = output_dir / "final" / "00000.jsonl.gz"
    assert final_output_file.exists() or (output_dir / "final" / "00000.jsonl").exists()

def test_run_data_prep_pipeline_fasttext(tmp_path, mocker):
    # Mock FastTextClassifierFilter to avoid downloading model during test
    from datatrove.pipeline.filters import FastTextClassifierFilter
    mocker.patch.object(FastTextClassifierFilter, '__init__', return_value=None)
    mocker.patch.object(FastTextClassifierFilter, '__call__', return_value=[])

    # Create dummy raw data
    input_dir = tmp_path / "data" / "raw"
    input_dir.mkdir(parents=True)
    raw_file = input_dir / "data.jsonl"
    with open(raw_file, "w") as f:
        f.write(json.dumps({"text": "Hello world", "id": "1"}) + "\n")

    output_dir = tmp_path / "data" / "deduplicated"
    minhash_dir = tmp_path / "data" / "minhash"

    cfg = {
        "input_path": str(input_dir),
        "output_path": str(output_dir),
        "minhash_base_path": str(minhash_dir),
        "minhash_config": {
            "n_grams": 2,
            "num_buckets": 2,
            "hashes_per_bucket": 2,
            "precision": 64
        },
        "filters": {
            "fasttext_spam": {
                "enabled": True,
                "model_url": "hf://test"
            },
            "fineweb_quality": {
                "enabled": False
            }
        }
    }

    run_data_prep_pipeline(cfg)


def test_run_data_prep_pipeline_transformers(tmp_path, mocker):

    # Mock pipeline factory
    mock_pipeline = mocker.MagicMock()
    # Mock the return value of calling the pipeline directly
    # Our dummy texts are "Hello world" and "Hello python"
    mock_pipeline.return_value = [
        {"label": "4", "score": 0.8},
        {"label": "2", "score": 0.4}
    ]

    # We must patch the pipeline function at the location it is imported in filters
    mocker.patch('src.data_prep.filters.pipeline', return_value=mock_pipeline)

    input_dir = tmp_path / "data" / "raw"
    input_dir.mkdir(parents=True)
    raw_file = input_dir / "data.jsonl"
    with open(raw_file, "w") as f:
        f.write(json.dumps({"text": "Hello world", "id": "1"}) + "\n")
        f.write(json.dumps({"text": "Hello python", "id": "2"}) + "\n")

    output_dir = tmp_path / "data" / "deduplicated"
    minhash_dir = tmp_path / "data" / "minhash"

    cfg = {
        "input_path": str(input_dir),
        "output_path": str(output_dir),
        "minhash_base_path": str(minhash_dir),
        "minhash_config": {
            "n_grams": 2,
            "num_buckets": 2,
            "hashes_per_bucket": 2,
            "precision": 64
        },
        "filters": {
            "fasttext_spam": {"enabled": False},
            "fineweb_quality": {"enabled": False},
            "transformers_classifier": {
                "enabled": True,
                "model_name": "dummy_model",
                "batch_size": 2,
                "keep_labels": [
                    ["4", 0.5],
                    ["5", 0.5]
                ],
                "device": "cpu"
            }
        }
    }

    run_data_prep_pipeline(cfg)

    final_output_file = output_dir / "final" / "00000.jsonl.gz"
    assert final_output_file.exists() or (output_dir / "final" / "00000.jsonl").exists()

def test_transformers_classifier_remove_labels(mocker):
    from src.data_prep.filters import TransformersClassifierFilter
    from datatrove.data import Document

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [
        {"label": "spam", "score": 0.9},
        {"label": "ham", "score": 0.8}
    ]
    mocker.patch('src.data_prep.filters.pipeline', return_value=mock_pipeline)

    filter_obj = TransformersClassifierFilter(
        model_name="dummy",
        remove_labels=[("spam", 0.5)],
        device="cpu"
    )

    docs = [
        Document(text="spam text", id="1"),
        Document(text="ham text", id="2")
    ]

    results = filter_obj.filter_batch(docs)

    assert not results[0][0]
    assert results[1]

def test_transformers_classifier_keep_labels_list(mocker):
    from src.data_prep.filters import TransformersClassifierFilter
    from datatrove.data import Document

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [
        {"label": "4", "score": 0.9},
        {"label": "3", "score": 0.8}
    ]
    mocker.patch('src.data_prep.filters.pipeline', return_value=mock_pipeline)

    filter_obj = TransformersClassifierFilter(
        model_name="dummy",
        keep_labels=("4", 0.8), # Pass as tuple to test tuple -> list conversion
        device="cpu"
    )

    docs = [
        Document(text="high quality", id="1"),
        Document(text="low quality", id="2")
    ]

    results = filter_obj.filter_batch(docs)

    assert results[0]
    assert not results[1][0]
