import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.config import project_path


DEFAULT_INPUT_PATH = project_path("data/processed/cfpb_sample_90k_clean.csv")
DEFAULT_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DEFAULT_MODEL_CACHE_DIR = project_path("artifacts/summarization/models")


@dataclass(frozen=True)
class SummaryConfig:
    model_name: str = DEFAULT_MODEL_NAME
    model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR
    text_column: str = "text_transformer"
    max_input_tokens: int = 1024
    max_summary_tokens: int = 110
    min_summary_tokens: int = 35
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    local_files_only: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CFPB complaint narratives.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--complaint-id", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-input-tokens", type=int, default=SummaryConfig.max_input_tokens)
    parser.add_argument("--max-summary-tokens", type=int, default=SummaryConfig.max_summary_tokens)
    parser.add_argument("--min-summary-tokens", type=int, default=SummaryConfig.min_summary_tokens)
    parser.add_argument("--local-files-only", action="store_true", help="Load the model from the local cache without network checks.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cached_snapshot_path(model_name: str, cache_dir: Path) -> Path | None:
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_cache / "refs" / "main"
    if not ref_path.exists():
        return None
    snapshot_id = ref_path.read_text(encoding="utf-8").strip()
    snapshot_path = repo_cache / "snapshots" / snapshot_id
    return snapshot_path if snapshot_path.exists() else None


def model_source(config: SummaryConfig) -> str:
    if config.local_files_only:
        path = Path(config.model_name)
        if path.exists():
            return str(path)
        snapshot_path = cached_snapshot_path(config.model_name, config.model_cache_dir)
        if snapshot_path is not None:
            return str(snapshot_path)
    return config.model_name


def load_complaint_by_id(input_path: Path, complaint_id: str, text_column: str) -> pd.Series:
    df = pd.read_csv(resolve_path(input_path), low_memory=False)
    matches = df[df["Complaint ID"].astype(str) == str(complaint_id)]
    if matches.empty:
        raise ValueError(f"Complaint ID not found: {complaint_id}")
    row = matches.iloc[0]
    if pd.isna(row.get(text_column)):
        raise ValueError(f"Complaint ID {complaint_id} has no text in column {text_column}")
    return row


class ComplaintSummarizer:
    def __init__(self, config: SummaryConfig = SummaryConfig()):
        self.config = config
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        source = model_source(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            cache_dir=str(config.model_cache_dir),
            local_files_only=config.local_files_only,
        )
        self.device = detect_device()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            source,
            cache_dir=str(config.model_cache_dir),
            local_files_only=config.local_files_only,
            use_safetensors=False,
        ).to(self.device)
        if self.model.generation_config.forced_bos_token_id is None and self.tokenizer.bos_token_id is not None:
            self.model.generation_config.forced_bos_token_id = self.tokenizer.bos_token_id
        self.model.eval()

    def truncate_text(self, text: str) -> str:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_input_tokens,
            return_tensors=None,
        )
        return self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    def summarize(self, text: str) -> dict:
        text = str(text).strip()
        if not text:
            raise ValueError("Cannot summarize empty text.")

        truncated = self.truncate_text(text)
        inputs = self.tokenizer(
            truncated,
            truncation=True,
            max_length=self.config.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=self.config.max_summary_tokens,
                min_length=self.config.min_summary_tokens,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=False,
                early_stopping=True,
            )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        input_words = len(text.split())
        summary_words = len(summary.split())
        return {
            "summary": summary,
            "input_word_count": input_words,
            "summary_word_count": summary_words,
            "compression_ratio": round(summary_words / input_words, 4) if input_words else 0,
            "was_truncated": text != truncated,
            "model_name": self.config.model_name,
        }


def print_summary(result: dict, metadata: dict | None = None) -> None:
    if metadata:
        print(json.dumps(metadata, indent=2, default=str))
        print()
    print("Summary:")
    print(result["summary"])
    print()
    print(f"Input words: {result['input_word_count']}")
    print(f"Summary words: {result['summary_word_count']}")
    print(f"Compression ratio: {result['compression_ratio']}")
    print(f"Was truncated: {result['was_truncated']}")


def main() -> None:
    args = parse_args()
    config = SummaryConfig(
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
        max_summary_tokens=args.max_summary_tokens,
        min_summary_tokens=args.min_summary_tokens,
        local_files_only=args.local_files_only,
    )

    if not args.complaint_id and not args.text:
        raise SystemExit("Provide either --complaint-id or --text.")

    if args.complaint_id:
        row = load_complaint_by_id(args.input, args.complaint_id, config.text_column)
        text = row[config.text_column]
        metadata = {
            "Complaint ID": row.get("Complaint ID"),
            "Product": row.get("Product"),
            "Issue": row.get("Issue"),
            "Company": row.get("Company"),
        }
    else:
        text = args.text
        metadata = None

    summarizer = ComplaintSummarizer(config)
    result = summarizer.summarize(text)
    print_summary(result, metadata)


if __name__ == "__main__":
    main()
