# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "absl-py",
#     "etils[eapp,edc,epath,etqdm]",
#     "beartype",
#     "jinja2",
#     "toolz",
#     "clu",
#     "huggingface-hub",
#     "grain",
#     "safetensors",
#     "numpy",
#     "vllm>=0.7",
#     "datasets>=3.2.0",
#     "polars",
# ]
# ///


import abc
import dataclasses
from itertools import chain
from typing import Any, Callable, Sequence

import polars as pl
import toolz as tlz
from absl import app, logging
from beartype import beartype
from datasets import Dataset, load_dataset
from etils import eapp, edc, epath, etqdm
from simple_parsing import Serializable, field, subgroups
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import Logprob


@dataclasses.dataclass
class SplitDatasetConfig(abc.ABC, Serializable):
    path: str
    split: str
    batch_size: int = 1


@edc.dataclass
@dataclasses.dataclass
class TrainSplitDatasetConfig(SplitDatasetConfig):
    split: str = "train"


@edc.dataclass
@dataclasses.dataclass
class TestSplitDatasetConfig(SplitDatasetConfig):
    split: str = "test"


@edc.dataclass
@dataclasses.dataclass
class ValSplitDatasetConfig(SplitDatasetConfig):
    split: str = "val"


@dataclasses.dataclass
class DatasetConfig(abc.ABC, Serializable):
    name: str
    train: TrainSplitDatasetConfig
    val: ValSplitDatasetConfig | None = None
    test: TestSplitDatasetConfig | None = None
    num_proc: int | None = None

    @abc.abstractmethod
    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        pass


@edc.dataclass
@dataclasses.dataclass
class MrQaDatasetConfig(DatasetConfig):
    name: str = "mrqua"
    train: TrainSplitDatasetConfig = field(default_factory=lambda: TrainSplitDatasetConfig(path="mrqa"))


@edc.dataclass
@dataclasses.dataclass
class SquadDatasetConfig(DatasetConfig):
    name: str = "squad"
    train: TrainSplitDatasetConfig = field(default_factory=lambda: TrainSplitDatasetConfig(path="squad"))


@edc.dataclass
@dataclasses.dataclass
class NaturalQADatasetConfig(DatasetConfig):
    name: str = "naturalqa"
    train: TrainSplitDatasetConfig = field(
        default_factory=lambda: TrainSplitDatasetConfig(path="google-research-datasets/natural_questions")
    )

    def sample_fn(self, sample: dict[str, Any]) -> dict[str, Any]:
        question = tlz.get_in(["question", "text"], sample, no_default=True)
        short_answers = tlz.get_in(["annotations", "short_answers"], sample, no_default=True)
        texts = tlz.pluck("text", short_answers)
        all_texts = chain.from_iterable(texts)
        answer = " ".join(all_texts)
        if not answer:
            return {"question": None, "answer": None}
        return {"question": question, "answer": answer}


@edc.dataclass
@dataclasses.dataclass
class SamplingConfig:
    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    ignore_eos: bool = False
    max_tokens: int = 1024
    min_tokens: int = 0
    logprobs: int | None = 0
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE


@edc.dataclass
@dataclasses.dataclass
class ModelConfig:
    model: str
    task: str = "generate"
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False


@edc.dataclass
@dataclasses.dataclass
class MistralModelConfig(ModelConfig):
    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    model: ModelConfig = subgroups({"default": ModelConfig, "mistral": MistralModelConfig}, default="default")
    dataset: DatasetConfig = subgroups(
        {"mrqa": MrQaDatasetConfig, "squad": SquadDatasetConfig, "naturalqa": NaturalQADatasetConfig}, default="squad"
    )
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")
    sampling_params: SamplingConfig = field(default_factory=SamplingConfig)
    limit: int | None = None


@beartype
def make_dataset(
    config: SplitDatasetConfig, sample_fn: Callable[[dict[str, Any]], dict[str, Any]], num_proc: int | None
) -> Dataset:
    ds = load_dataset(config.path, split=config.split)
    ds = ds.map(sample_fn, num_proc=num_proc)
    ds = ds.filter(lambda x: x["answer"] is not None, num_proc=num_proc)
    if config.batch_size > 1:
        ds = ds.batch(config.batch_size, num_proc=num_proc)
    return ds


@beartype
def extract_logprobs(logprobs: Sequence[dict[int, Logprob]]) -> tuple[Sequence[int], Sequence[float]]:
    tokens, lps = zip(*chain.from_iterable([lp.items() for lp in logprobs]))
    return (tokens, tuple(lp.logprob for lp in lps))


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug("Make dataset")
    ds = make_dataset(cfg.dataset.train, cfg.dataset.sample_fn, num_proc=cfg.dataset.num_proc)
    sampling_params = SamplingParams(**dataclasses.asdict(cfg.sampling_params))
    logging.debug("Make llm")
    llm = LLM(**dataclasses.asdict(cfg.model))
    output_file = cfg.output_dir / f"responses_{cfg.model.model}_{cfg.dataset.name}.json".replace("/", "_")
    logging.debug("Output file %s", output_file)
    if output_file.exists():
        msg = f"{output_file} already exists"
        raise FileExistsError(msg)
    with output_file.open("a") as dst:
        for batch in etqdm.tqdm(ds):
            requests = llm.generate(batch["question"], sampling_params)
            responses, logprobs = zip(*((req.outputs[0].text, req.outputs[0].logprobs) for req in requests))
            tokens, processed = zip(*map(extract_logprobs, logprobs))
            sample = {
                "id": batch["id"],
                "response": responses,
                "question": batch["question"],
                "answer": batch["answer"],
                "tokens": tokens,
                "logprobs": processed,
            }

            df = pl.DataFrame(sample)
            df.write_ndjson(dst)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
