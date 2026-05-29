import os
import json
from src.data_prep.pipeline import run_data_prep_pipeline
import tempfile
import yaml

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
