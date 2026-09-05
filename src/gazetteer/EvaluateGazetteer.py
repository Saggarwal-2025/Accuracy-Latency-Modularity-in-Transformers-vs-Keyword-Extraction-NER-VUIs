import json
import re
import time
import statistics
import argparse
from dataclasses import dataclass
import glob
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "splits"
FILLER_VOCAB = ["um", "uh", "eh"]
GENERIC_STOPWORDS = ["please", "can", "you", "okay", "now", "just"]

CUSTOM_STOPWORDS = FILLER_VOCAB + GENERIC_STOPWORDS

ENTITY_CLASSES = ["action", "target", "correction_connector"]


# method to load vocab exported from ExportVocab.py
def _load_vocab(path: str) -> dict:
    with open(path, "r") as f:
        vocab = json.load(f)

    for class_name in ENTITY_CLASSES:
        if class_name not in vocab:
            raise KeyError(f"vocab.json is missing required class named {class_name}")

    return vocab


def _dataset_config(dataset: str) -> tuple[list[str], list[str], list[str]]:
    if dataset == "closed":
        root = DATA_ROOT / "closed"
        data_paths = [str(root / "simple" / "*.json"), str(root / "complex" / "*.json")]
        vocab_paths = data_paths
    else:
        root = DATA_ROOT / "open"
        data_paths = [
            str(root / "vocab_test" / "simple" / "test" / "*.json"),
            str(root / "vocab_test" / "complex" / "test" / "*.json"),
        ]
        vocab_paths = [
            str(root / "vocab_train" / "simple" / "train" / "*.json"),
            str(root / "vocab_train" / "complex" / "train" / "*.json"),
        ]
    return data_paths[:1], data_paths[1:], vocab_paths


def _load_vocab_from_dataset(paths: list[str]) -> dict:
    examples = []
    for path in paths:
        for file_path in sorted(glob.glob(path)):
            with open(file_path, "r") as f:
                examples.extend(json.load(f)["rasa_nlu_data"]["common_examples"])
    return {
        class_name: sorted(
            {
                entity["value"]
                for example in examples
                for entity in example.get("entities", [])
                if entity.get("entity") == class_name
            }
        )
        for class_name in ENTITY_CLASSES
    }


def _compile_vocab(vocab: dict) -> dict:
    return {
        class_name: [
            (entry, re.compile(r"(?<!\w)" + re.escape(entry.lower()) + r"(?!\w)"))
            for entry in sorted(entries, key=len, reverse=True)
        ]
        for class_name, entries in vocab.items()
    }


@dataclass
class Example:
    sentence: str
    actual_labels: dict
    complexity: str = "simple"
    has_filler: bool = False


FILLER_VOCAB = ["um", "uh", "eh"]


def _entities_to_actual_labels(entities: list) -> dict:
    labels = {class_name: None for class_name in ENTITY_CLASSES}
    for entity in entities:
        entity_class = entity["entity"]
        if entity_class in labels:
            labels[entity_class] = entity["value"]  # last occurrence wins
    return labels


# method to load each of chatette's generated rasa nlu json output directly for each complexity type separately - enlisted the help of Gemini
def load_dataset(paths, complexity_label: str) -> list[Example]:
    if isinstance(paths, str):
        file_list = sorted(glob.glob(paths))
    elif isinstance(paths, (list, tuple)):
        file_list = []
        for path_or_pattern in paths:
            if isinstance(path_or_pattern, str) and glob.has_magic(path_or_pattern):
                file_list.extend(sorted(glob.glob(path_or_pattern)))
            else:
                file_list.append(path_or_pattern)
        file_list = sorted(file_list)
    else:
        file_list = sorted(paths)

    if not file_list:
        raise FileNotFoundError(f"No files found matching {paths}")

    common_examples = []

    for path in file_list:
        with open(path, "r") as f:
            raw = json.load(f)
        common_examples.extend(raw["rasa_nlu_data"]["common_examples"])

    examples = []

    for example in common_examples:
        text = example["text"]
        entities = example.get("entities", [])
        examples.append(
            Example(
                sentence=text,
                actual_labels=_entities_to_actual_labels(entities),
                complexity=complexity_label,
                has_filler=any(w in text.lower().split() for w in FILLER_VOCAB),
            )
        )

    return examples


# method to run pure gazetteer on data without ranking step, based on method from EvaluateRAKE.py
def run_gazetteer_on_example(sentence: str, compiled_vocab: dict) -> tuple[dict, float]:
    start = time.perf_counter()

    sentence_l = sentence.lower()
    predicted = {class_name: None for class_name in ENTITY_CLASSES}
    # tracking the start index of the current best (rightmost) match per class
    best_start = {class_name: -1 for class_name in ENTITY_CLASSES}

    for class_name, compiled_entries in compiled_vocab.items():
        for entry, pattern in compiled_entries:
            for match in pattern.finditer(sentence_l):
                if match.start() > best_start[class_name]:
                    best_start[class_name] = match.start()
                    predicted[class_name] = entry

    elapsed_ms = (time.perf_counter() - start) * 1000
    return predicted, elapsed_ms


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate(examples: list[Example], predictions: list[dict]) -> dict:
    results = {
        "overall": {},
        "simple": {},
        "complex": {},
        "filler_present": {},
        "filler_absent": {},
    }
    buckets = {
        "overall": lambda ex: True,
        "simple": lambda ex: ex.complexity == "simple",
        "complex": lambda ex: ex.complexity == "complex",
        "filler_present": lambda ex: ex.has_filler,
        "filler_absent": lambda ex: not ex.has_filler,
    }
    for entity in ENTITY_CLASSES:
        per_bucket_true = {name: [] for name in buckets}
        per_bucket_pred = {name: [] for name in buckets}
        for ex, pred in zip(examples, predictions):
            gt = str(ex.actual_labels.get(entity))
            pd_ = str(pred.get(entity))
            for name, keep in buckets.items():
                if keep(ex):
                    per_bucket_true[name].append(gt)
                    per_bucket_pred[name].append(pd_)
        for name in buckets:
            if per_bucket_true[name]:
                results[name][entity] = compute_metrics(
                    per_bucket_true[name], per_bucket_pred[name]
                )
    return results


# main to run everything together, taken from EvaluateRAKE and changed words within print statements
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("closed", "open"), default="closed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    simple_paths, complex_paths, vocab_paths = _dataset_config(args.dataset)
    vocab = _load_vocab_from_dataset(vocab_paths)
    compiled_vocab = _compile_vocab(vocab)

    print("Loading datasets...")
    simple_examples = load_dataset(simple_paths, "simple")
    complex_examples = load_dataset(complex_paths, "complex")
    examples = simple_examples + complex_examples
    print(
        f"Loaded {len(simple_examples)} simple / {len(complex_examples)} complex "
        f"({len(examples)} total)"
    )

    print("Running pure gazetteer matching...")
    predictions = []
    for ex in examples:
        pred, _ = run_gazetteer_on_example(ex.sentence, compiled_vocab)
        predictions.append(pred)

    print("Computing metrics...")
    metrics = evaluate(examples, predictions)

    print("Getting latency for 5 passes...")
    all_latencies = []
    for _ in range(5):
        for ex in examples:
            _, ms = run_gazetteer_on_example(ex.sentence, compiled_vocab)
            all_latencies.append(ms)
    latency = {
        "mean_ms": statistics.mean(all_latencies),
        "median_ms": statistics.median(all_latencies),
        "stdev_ms": statistics.stdev(all_latencies),
        "n_samples": len(all_latencies),
    }

    output = {"metrics": metrics, "latency": latency}
    output_path = args.output or REPO_ROOT / f"gazetteer_{args.dataset}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    main()
