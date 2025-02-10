# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "etils[eapp,edc,epath]",
#     "matplotib",
#     "polars",
#     "scikit-learn",
#     "toolz",
# ]
# ///
"""
Script that takes a responses and a score file.
It read samples from both file, join them on the id then plot calibration curve using the score key.
If there is no score key, it uses the logprobs key to compute a naive score.
If here is no logprobs key, it returns an error
"""

import dataclasses

import orjson as json
import polars as pl
from absl import app, logging
from etils import eapp, edc, epath
from sklearn.calibration import calibration_curve

import wandb


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    ratings_files: epath.Path
    threshold: int
    name: str | None = None
    n_bins: int = 10
    random_seed: int = 42
    x_label: str = "Mean predicted value"
    y_label: str = "Fraction of positives"


def main(cfg: AppConfig):
    wandb.init(config=dataclasses.asdict(cfg), project="artefactual", name=cfg.name)
    logging.info("\n%s", cfg)
    with cfg.ratings_files.open("r") as src:
        samples = map(json.loads, src)
        df = pl.DataFrame(samples)

    logging.info("Read %d rows", len(df))
    # Remove missing ratings and convert ratings to int
    df = df.filter(pl.col("rating").is_not_null()).with_columns([
        pl.col("rating").cast(pl.Float32, strict=False).cast(pl.UInt16, strict=False)
    ])
    y_true = df.select(pl.col("rating") > cfg.threshold).cast(pl.UInt8).to_numpy().flatten()

    logging.info("Plot %d", len(df))

    if "score" not in df:
        logging.warning("Compute naive score")
        # Compute a naive score if score is missing
        df = df.with_columns(pl.col("logprobs").list.mean().exp().alias("score"))

    y_probas = df["score"].to_numpy()
    frac_pos, pred_values = calibration_curve(y_true, y_probas, n_bins=cfg.n_bins, pos_label=True)
    data_line = list(zip(frac_pos, pred_values))
    table_line = wandb.Table(data=data_line, columns=[cfg.x_label, cfg.y_label])
    # hist, edges = np.histogram(y_probas, bins=cfg.n_bins, density=False)
    data_hist = [[h] for h in y_probas]
    breakpoint()
    table_data = wandb.Table(data=data_hist, columns=["Score"])

    wandb.log({
        "calibration_plot": wandb.plot.line(table_line, cfg.x_label, cfg.y_label, title="Calibration plot"),
        "histogram": wandb.plot.histogram(table_data, "Score", title="Score distribution"),
    })


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
