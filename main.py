import click
import yaml
import tempfile
import fasttext
from datasets import Dataset, interleave_datasets, load_dataset
from src.tokenizer import build_tokenizer, run_lexical_initialization
from src.pruning import prune_tokenizer_and_model

from transformers import AutoTokenizer
from src.data_prep.pipeline import run_data_prep_pipeline
from src.cpt import run_cpt
from src.sft import run_sft
from src.alignment import run_alignment_pipeline
from src.task_sft import run_task_sft

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """CLI for the Language Model Training Pipeline"""
    pass

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to tokenizer configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def tokenizer(config, dummy_data):
    """Run Tokenizer Training & FOCUS stage."""
    cfg = load_config(config)
    click.echo(f"Starting Tokenizer stage with config: {cfg}")

    if dummy_data:
        click.echo("Using simple dummy list for tokenizer training...")
        data_iterator = ["print('Hello')", "12345", "кошка сидит", "neural network"]
    else:
        click.echo("Streaming datasets based on config ratios and interleaving...")
        ratios = cfg.get("dataset_ratios", {"ru": 0.40, "en": 0.30, "code": 0.15, "math": 0.15})
        paths = cfg.get("dataset_paths", {})

        datasets = []
        probabilities = []

        for lang, ratio in ratios.items():
            path = paths.get(lang)
            if not path:
                click.echo(f"Warning: No dataset path provided for '{lang}'. Skipping.")
                continue

            # Simple assumption: using dataset with "text" column
            try:
                ds = load_dataset(path, streaming=True, split="train")
                datasets.append(ds)
                probabilities.append(ratio)
            except Exception as e:
                click.echo(f"Error loading {path}: {e}")

        if not datasets:
            click.echo("Error: No datasets loaded successfully. Aborting.")
            return

        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]

        combined_ds = interleave_datasets(datasets, probabilities=probabilities, stopping_strategy="all_exhausted")

        def text_iterator():
            count = 0
            # limit strictly for demonstration or add a param to config for 'max_samples'
            limit = cfg.get("max_samples", 100000)
            for item in combined_ds:
                if count > limit: break
                if "text" in item and item["text"]:
                    yield item["text"]
                    count += 1
                elif "content" in item and item["content"]:
                    yield item["content"]
                    count += 1

        # We materialize it here so we can pass it to fasttext as well.
        # For huge datasets we'd write it to disk instead of RAM.
        data_iterator = list(text_iterator())

    # 1. Train Tokenizer
    vocab_size = cfg.get("vocab_size", 5000)
    click.echo(f"Training new tokenizer (BPE + Digits splitting) to size {vocab_size}...")
    new_tok = build_tokenizer(data_iterator, vocab_size=vocab_size)
    new_tok.save("new_tokenizer.json")

    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    base_tok = AutoTokenizer.from_pretrained(model_name)
    base_vocab = set(base_tok.get_vocab().keys())

    new_vocab = new_tok.get_vocab().keys()

    # Extract strictly new tokens
    new_ru_tokens = [tok for tok in new_vocab if tok not in base_vocab and not tok.startswith("<")]

    if len(new_ru_tokens) == 0:
        click.echo("Warning: No strictly new tokens found. Ensure you are training on new data.")
        # fallback for dummy testing
        new_ru_tokens = ["Ġкошка", "Ġсобака"]

    click.echo(f"Found {len(new_ru_tokens)} new tokens to initialize.")

    ft_model_path = None
    if cfg.get("focus", False):
        click.echo("Training auxiliary fastText model for FOCUS...")
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            for text in data_iterator:
                f.write(text + "\n")
            tmp_name = f.name

        ft_model = fasttext.train_unsupervised(tmp_name, model='cbow', dim=32, epoch=5, minCount=1)
        ft_model_path = tmp_name + ".bin"
        ft_model.save_model(ft_model_path)

    # 2. Run Lexical Initialization
    click.echo(f"Running Lexical Initialization on model {model_name}...")

    # we can pass save_path from config or default it to None so we don't pollute repo during test
    save_path = cfg.get("save_path", None)
    run_lexical_initialization(model_name, new_ru_tokens, cfg, fasttext_model_path=ft_model_path, save_path=save_path)

    click.echo("Tokenizer stage completed successfully.")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to pruning configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def prune(config, dummy_data):
    """Run Tokenizer Pruning & Optimization stage."""
    cfg = load_config(config)
    click.echo(f"Starting Pruning stage with config: {cfg}")

    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    min_freq = cfg.get("min_freq", 100)
    output_dir = cfg.get("output_dir", "./lfm-pruned")

    if dummy_data:
        click.echo("Using simple dummy dataset for pruning...")
        dataset = [{"text": "hello world"}, {"text": "hello python"}, {"text": "кошка"}]
    else:
        # Load dataset according to your actual config logic
        dataset_path = cfg.get("dataset_path")
        if dataset_path:
            click.echo(f"Loading dataset from {dataset_path}...")
            dataset = load_dataset("json", data_files=f"{dataset_path}/*.jsonl", split="train")
            max_samples = cfg.get("max_samples", 500000)
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        else:
            click.echo("Error: No dataset_path provided and not using dummy data.")
            return

    prune_tokenizer_and_model(model_name, dataset, min_freq, output_dir)
    click.echo("Pruning stage completed successfully.")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to data prep configuration YAML.')
def data_prep(config):
    """Run Data Deduplication (MinHash LSH) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Data Prep stage with config: {cfg}")
    run_data_prep_pipeline(cfg)

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to CPT configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def cpt(config, dummy_data):
    """Run Continual Pre-Training (CPT) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Continual Pre-Training (CPT) stage with config: {cfg}")
    run_cpt(cfg, dummy_data)

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to SFT configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def sft(config, dummy_data):
    """Run Supervised Fine-Tuning (General SFT) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Supervised Fine-Tuning (General SFT) stage with config: {cfg}")
    run_sft(cfg, dummy_data)

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to alignment configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def alignment(config, dummy_data):
    """Run Alignment (DPO/GRPO) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Alignment stage with config: {cfg}")
    run_alignment_pipeline(cfg, dummy_data)

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to task SFT configuration YAML.')
@click.option('--dummy-data', is_flag=True, help='Use simple hardcoded list for fast testing.')
def task_sft(config, dummy_data):
    """Run Final Task SFT stage."""
    cfg = load_config(config)
    click.echo(f"Starting Final Task SFT stage with config: {cfg}")
    run_task_sft(cfg, dummy_data)

if __name__ == '__main__':
    cli()
