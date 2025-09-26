from pathlib import Path

import click
import yaml


def load_config(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """MitoChime CLI"""
    pass


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def align(config):
    cfg = load_config(config)
    Path(cfg["paths"]["outdir"]).mkdir(parents=True, exist_ok=True)
    click.echo(
        f"[align] {cfg['paths']['reads_r1']} + {cfg['paths']['reads_r2']} "
        f"-> {cfg['paths']['refs_mito']} (stub)"
    )


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def features(config):
    Path("features").mkdir(exist_ok=True)
    click.echo("[features] (stub) -> features/*.tsv")


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def train(config):
    Path("models/xgb").mkdir(parents=True, exist_ok=True)
    click.echo("[train] (stub) -> models/xgb/mito_xgb_model.json")


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def predict(config):
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    click.echo("[predict] (stub) -> results/predictions/*.tsv")


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def eval(config):
    Path("results/figs").mkdir(parents=True, exist_ok=True)
    click.echo("[eval] (stub) -> results/figs/")


if __name__ == "__main__":
    cli()
