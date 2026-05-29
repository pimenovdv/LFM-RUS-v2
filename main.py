import click
import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """CLI for the Language Model Training Pipeline"""
    pass

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to tokenizer configuration YAML.')
def tokenizer(config):
    """Run Tokenizer Training & FOCUS stage."""
    cfg = load_config(config)
    click.echo(f"Starting Tokenizer stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to pruning configuration YAML.')
def prune(config):
    """Run Tokenizer Pruning & Optimization stage."""
    cfg = load_config(config)
    click.echo(f"Starting Pruning stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to data prep configuration YAML.')
def data_prep(config):
    """Run Data Deduplication (MinHash LSH) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Data Prep stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to CPT configuration YAML.')
def cpt(config):
    """Run Continual Pre-Training (CPT) stage."""
    cfg = load_config(config)
    click.echo(f"Starting CPT stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to SFT configuration YAML.')
def sft(config):
    """Run Supervised Fine-Tuning (General SFT) stage."""
    cfg = load_config(config)
    click.echo(f"Starting General SFT stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to alignment configuration YAML.')
def alignment(config):
    """Run Alignment (DPO/GRPO) stage."""
    cfg = load_config(config)
    click.echo(f"Starting Alignment stage with config: {cfg}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to task SFT configuration YAML.')
def task_sft(config):
    """Run Final Task SFT stage."""
    cfg = load_config(config)
    click.echo(f"Starting Task SFT stage with config: {cfg}")

if __name__ == '__main__':
    cli()
