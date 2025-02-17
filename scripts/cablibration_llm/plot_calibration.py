# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "etils[eapp,edc,epath]",
#     "orjson",
#     "polars",
#     "scikit-learn",
#     "wandb",
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
    scores_file: epath.Path
    name: str | None = None
    n_bins: int = 11
    random_seed: int = 42
    x_label: str = "Mean predicted value"
    y_label: str = "Fraction of positives"


def main(cfg: AppConfig):
    # TODO: Check if the calibration plot and the histograms are right.
    wandb.init(config=dataclasses.asdict(cfg), project="artefactual", name=cfg.name)
    logging.info("\n%s", cfg)
    with cfg.scores_file.open("r") as src:
        samples = map(json.loads, src)
        df = pl.DataFrame(samples).with_columns([pl.col("id").cast(pl.Int64, strict=False)])

    logging.info("Got %d rows", len(df))

    y_true = df.select(pl.col("label")).to_numpy().flatten()
    y_probas = df.select(pl.col("score")).to_numpy().flatten()
    frac_pos, pred_values = calibration_curve(y_true, y_probas, n_bins=cfg.n_bins, pos_label=True)
    data_line = list(zip(frac_pos, pred_values))
    table_line = wandb.Table(data=data_line, columns=[cfg.x_label, cfg.y_label])
    # hist, edges = np.histogram(y_probas, bins=cfg.n_bins, density=False)
    data_hist = [[h] for h in y_probas]
    table_data = wandb.Table(data=data_hist, columns=["Score"])

    wandb.log({
        "calibration_plot": wandb.plot.line(table_line, cfg.x_label, cfg.y_label, title="Calibration plot"),
        "histogram": wandb.plot.histogram(table_data, "Score", title="Score distribution"),
    })


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
