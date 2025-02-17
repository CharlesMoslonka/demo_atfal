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
#     "wandb",
#     "safetensors",
#     "numpy",
#     "vllm>=0.7",
#     "polars",
#     "tensorflow-datasets",
#     "orjson",
#     "outlines[vllm]",
#     "pydantic",
# ]
# ///


import dataclasses

import orjson as json
import polars as pl
import toolz as tlz
from absl import app, logging
from etils import eapp, edc, epath, etqdm
from jinja2 import Environment, Template
from outlines import generate, models
from pydantic import BaseModel, ValidationError
from simple_parsing import field, subgroups
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


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
    seed: int | None = 42
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    ignore_eos: bool = False
    max_tokens: int = 1024
    min_tokens: int = 0
    logprobs: int | None = None
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
    max_model_len: int = 8192


@edc.dataclass
@dataclasses.dataclass
class MistralModelConfig(ModelConfig):
    tokenizer_mode: str = "mistral"
    config_format: str = "mistral"
    load_format: str = "mistral"


# TODO: Ratings perf are still bad. Need to find a better prompt
DEFAULT_PROMPT = """
You are tasked with evaluating the correctness of a generated answer with respect to a golden answer.
Here are some rules of the evaluation:
You must read and understand the question being asked.
Your reply should strictly describe how similar the generated answer is to the golden answer.
If the Generated Answer contradicts any key fact in the Golden Answer, consider it a major error.
If the Generated Answer only differs slightly in wording or includes minor omissions but preserves correctness, \
        consider it a minor error.
If the Generated Answer is completely aligned with the Golden Answer, it should receive the highest score.
Assign a rating on a scale of 0 to 5:
- 0: Completely incorrect or irrelevant.
- 1: Mostly incorrect, with only slight correct elements.
- 2: Partially correct but contains major omissions or errors.
- 3: Mostly correct with some errors or missing details.
- 4: Correct with only minor issues or omissions.
- 5: Perfectly correct and fully consistent with the Golden Answer.

Question:
{{ question }}

Golden Answer:
{{ golden_answer }}

Generated AnsWer
{{ generated_answer }}
"""

# DEFAULT_PROMPT = """You are an impartial and strict evaluator. \
#        You must compare the "Generated Answer" to the "Golden Answer" for the given "Question" \
# and decide how closely the Generated Answer matches it in terms of correctness and completeness.
#
# Evaluation Rules:
# (1) Read the Question to understand what is being asked.
# (2) Review the Golden Answer. Assume it is correct and factual and use it as your reference point.
# (3) Compare the Generated Answer to the Golden Answer:
#    - If the Generated Answer contradicts any key fact in the Golden Answer, consider it a major error.
#    - If the Generated Answer only differs slightly in wording or includes minor omissions but preserves correctness, \
#        consider it a minor error.
#    - If the Generated Answer is completely aligned with the Golden Answer, it should receive the highest score.
# (4) Assign a rating on a scale of 0 to 5:
#    - 0: Completely incorrect or irrelevant.
#    - 1: Mostly incorrect, with only slight correct elements.
#    - 2: Partially correct but contains major omissions or errors.
#    - 3: Mostly correct with some errors or missing details.
#    - 4: Correct with only minor issues or omissions.
#    - 5: Perfectly correct and fully consistent with the Golden Answer.
# (5) Provide a concise explanation (1-3 sentences) describing your reasoning. Reference any critical discrepancies or confirmations with the Golden Answer.
#
# Question:
# {{ question }}
#
# Golden Answer:
# {{ golden_answer }}
#
# Generated AnsWer
# {{ generated_answer }}
#
# Rating:
# """
DEFAULT_PROMPT = """You are an impartial and strict evaluator. Your goal is to compare the "Generated Answer" to the authoritative "Golden Answer" for a given question and decide how closely the Generated Answer matches it in terms of correctness and completeness.

**Instructions**:

1. **Read the Question** carefully to understand the context.

2. **Review the Golden Answer**. This is the official, correct answer. Assume it is factually accurate and use it as your reference point.

3. **Examine the Generated Answer**. Identify whether it:
   - Matches the main facts stated in the Golden Answer (dates, names, definitions, etc.).
   - Conflicts with or omits crucial information from the Golden Answer.
   - Adds any relevant or irrelevant details.

4. **Compare for factual alignment**:
   - If the Generated Answer contradicts any key fact in the Golden Answer, consider it a major error.
   - If the Generated Answer only differs slightly in wording or includes minor omissions but preserves correctness, consider it a minor error.
   - If the Generated Answer is completely aligned with the Golden Answer, it should receive the highest score.

5. **Assign a rating (0 to 5)** using these criteria:
   - **0**: Completely incorrect or unrelated to the question.
   - **1**: Mostly incorrect, with minimal correct information or heavy factual conflicts.
   - **2**: Partially correct but missing significant details or containing major errors.
   - **3**: Mostly correct but with notable errors or omissions.
   - **4**: Correct with only minor factual discrepancies or omissions.
   - **5**: Perfectly correct, with no factual differences from the Golden Answer.

6. **Explain your reasoning** concisely:
   - Specify key points of agreement or disagreement.
   - Reference the Golden Answer facts when pointing out errors or accuracies.
---
<question>{{ question }}</question>
<golden_answer>{{ golden_answer }}</golden_answer>
<generated_answer>{{ generated_answer }}</generated_answer>
<think>
"""


@edc.dataclass
@dataclasses.dataclass
class PromptConfig:
    prompt: str | None = None
    template: Template = field(init=False)

    def __post_init__(self):
        if self.prompt is None:
            self.prompt = DEFAULT_PROMPT

        env = Environment(autoescape=True)
        self.template = env.from_string(self.prompt)

    def render(self, question: str, golden_answer: str, generated_answer: str) -> str:
        return self.template.render(question=question, golden_answer=golden_answer, generated_answer=generated_answer)


@edc.dataclass
@dataclasses.dataclass
class AppConfig:
    source: epath.Path
    model: ModelConfig = subgroups({"default": ModelConfig, "mistral": MistralModelConfig}, default="default")
    prompt: PromptConfig = edc.field(default_factory=PromptConfig)
    output_dir: epath.Path = edc.field(validate=epath.Path, default="outputs")
    sampling_params: SamplingConfig = field(default_factory=SamplingConfig)
    limit: int | None = None
    batch_size: int = 64
    resume: bool = False


class Response(BaseModel):
    explanation: str
    rating: int


def main(cfg: AppConfig):
    logging.info("\n%s", cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sampling_params = SamplingParams(**dataclasses.asdict(cfg.sampling_params))
    logging.debug("Make Embedder")
    llm = LLM(**dataclasses.asdict(cfg.model))
    model = models.VLLM(llm)
    output_file = cfg.output_dir / f"scores_{cfg.model.model}_{cfg.source.name}.json".replace("/", "_")
    logging.debug("Output file %s", output_file)
    if output_file.exists():
        msg = f"{output_file} already exists"
        raise FileExistsError(msg)

    generator = generate.json(model, Response)

    output_file = cfg.output_dir / f"ratings_{cfg.model.model}_{cfg.source.name}".replace("/", "_")
    if output_file.exists():
        if not cfg.resume:
            msg = f"Output file {output_file} already exists"
            raise ValueError(msg)
        else:
            with output_file.open("r") as src:
                samples = map(json.loads, src)
                existing_ids = set(tlz.pluck("id", samples))
    else:
        existing_ids = {}

    with cfg.source.open("r") as src:
        with output_file.open("a") as dst:
            while etqdm.tqdm(src):
                lines = tlz.take(cfg.batch_size, src)
                samples = map(json.loads, lines)
                samples = tlz.filter(lambda s: s["id"] not in existing_ids, samples)
                try:
                    _, samples = tlz.peek(samples)
                except StopIteration:
                    break

                ids, questions, responses, answers = zip(
                    *tlz.pluck(["id", "question", "response", "answer"], samples), strict=True
                )
                prompts = [
                    cfg.prompt.render(question=question, golden_answer=golden_answer, generated_answer=generated_answer)
                    for question, golden_answer, generated_answer in zip(questions, answers, responses, strict=True)
                ]
                try:
                    requests = generator(prompts, sampling_params=sampling_params)
                except ValidationError:
                    msg = "Error validating generated output"
                    logging.exception(msg)
                    continue
                ratings, explanations = zip(*((req.rating, req.explanation) for req in requests), strict=False)
                samples = {"id": ids, "rating": ratings, "explanation": explanations}
                df = pl.DataFrame(samples)
                df.write_ndjson(dst)
    logging.info("Wrote results in %s", output_file)


if __name__ == "__main__":
    eapp.better_logging()
    raise SystemExit(app.run(main, flags_parser=eapp.make_flags_parser(AppConfig)))  # pyright: ignore[reportArgumentType]
