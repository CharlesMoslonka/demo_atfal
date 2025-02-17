# /// script
# requires-python = ">=3.11"

# dependencies = [
#     "einops",
#     "etils[eapp,edc,epath]",
#     "jaxtyping",
#     "numpy",
#     "orjson",
#     "plum-dispatch",
#     "polars",
#     "scikit-learn",
#     "toolz",
# ]
# ///
"""
Compute the score from a list of response provided by a llm

Must output id, scores
"""

import dataclasses
from collections.abc import Sequence
from enum import auto
from functools import partial
from typing import Any, Callable, Literal

import numpy as np
import orjson as json
import polars as pl
import tlz
from absl import app, logging
from beartype import beartype
from einops import rearrange, reduce
from etils import eapp, edc, epath
from etils.epy import StrEnum
from numpy.typing import NDArray
from plum import dispatch, overload
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class ScoringMethod(StrEnum):
    NAIVE = auto()  # https://cookbook.openai.com/examples/using_logprobs
    SUPERVISED_ISOTONIC = auto()
    SUPERVISED_SIGMOID = auto()

    @staticmethod
    def supervised():
        return {
            ScoringMethod.SUPERVISED_ISOTONIC,
            ScoringMethod.SUPERVISED_SIGMOID,
        }

    @staticmethod
    def unsupervised():
        return {ScoringMethod.NAIVE}


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    responses_file: epath.Path
    ratings_file: epath.Path
    max_length: int
    batch_size: int = 2**20
    output_dir: epath.Path = epath.Path("outputs")
    method: ScoringMethod = ScoringMethod.NAIVE
    threshold: int = 4


Logprobs = NDArray[np.float32]
Scores = NDArray[np.float32]


@beartype
def process_logprobs(logprobs: Sequence[Sequence[float]], max_len: int) -> Logprobs:
    logprobs = (lp[:max_len] for lp in logprobs)
    logprobs = [np.pad(lp, (0, max_len - len(lp)), mode="constant") for lp in logprobs]
    return np.array(logprobs, dtype=np.float32)


@overload
def score_fn(method: ScoringMethod, logprobs: Any):
    probs = np.exp(logprobs)
    scores = reduce(probs, "batch max_len -> batch", "mean")
    return scores


@overload
def score_fn(
    method: Literal[ScoringMethod.SUPERVISED_ISOTONIC, ScoringMethod.SUPERVISED_ISOTONIC],
    logprobs: Any,
    ids: Any,
    labels: Any,
):
    probs = np.exp(logprobs)
    scores = reduce(probs, "n_samples max_len -> n_samples", "mean")
    scores = rearrange(scores, "(n_samples dim) -> n_samples dim", dim=1)
    ids_train, ids_test, scores_train, scores_test, labels_train, labels_test = train_test_split(
        ids, scores, labels, test_size=0.8, random_state=42
    )
    if method is ScoringMethod.SUPERVSED_ISOTONIC:
        calibration_method = "isotonic"
    elif method is ScoringMethod.SUPERVSED_SIGMOID:
        calibration_method = "sigmoid"
    else:
        msg = f"method {method} is not recognized"
        raise ValueError(msg)
    calibrated = CalibratedClassifierCV(method=calibration_method)
    calibrated.fit(scores_train, labels_train)
    scores = calibrated.predict_proba(scores_test)
    return ids_test, scores[:, 0], labels_test


@dispatch
def score_fn(method: ScoringMethod, logprobs: Any, ids: Any | None, labels: Sequence[int] | None):
    raise NotImplementedError


@beartype
def read_file(path: epath.Path) -> list[dict[str, Any]]:
    with path.open("r") as src:
        lines = src.readlines()
        samples = map(json.loads, lines)
        return list(samples)


def join_samples(
    ratings: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    key_fn: Callable[[dict[str, str]], str] = tlz.curried.get("id"),
):
    joined = tlz.join(key_fn, responses, key_fn, ratings)
    return [{"id": left["id"], **left, **right} for left, right in joined]


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)

    output_files = cfg.output_dir / f"scores_{cfg.method}_{cfg.responses_file}".replace("/", "_")
    samples = read_file(cfg.responses_file)

    with output_files.open("w") as dst:
        rating_samples = tlz.pipe(cfg.ratings_file, read_file, tlz.curried.filter(lambda d: d["rating"] is not None))
        samples = join_samples(rating_samples, samples)
        ids, logprobs = zip(*tlz.pluck(["id", "logprobs"], samples))
        ids = np.array(list(map(int, ids)), dtype=int)

        ratings = tlz.pipe(samples, tlz.curried.pluck("rating"), tlz.curried.map(tlz.compose_left(float, int)))

        labels = tlz.pipe(
            ratings,
            tlz.curried.map(lambda rating: rating >= cfg.threshold),
            list,
            partial(np.array, dtype=int),
        )
        logprobs = process_logprobs(logprobs, max_len=cfg.max_length)

        if cfg.method in ScoringMethod.supervised():
            ids, scores, labels = score_fn(cfg.method, logprobs, ids, labels)
        elif cfg.method in ScoringMethod.unsupervised():
            scores = score_fn(cfg.method, logprobs)
        else:
            msg = "wrong method"
            raise ValueError(msg)
        df = pl.DataFrame({"id": ids, "score": scores, "label": labels})
        df.write_ndjson(dst)
    logging.info("Wrote scores into %s", output_files)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))
