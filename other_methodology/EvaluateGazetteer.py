import json
import re
import time
import statistics
from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

VOCAB_PATH = "vocab.json"
FILLER_VOCAB = ["um", "uh", "eh"]
GENERIC_STOPWORDS = ["please", "can", "you", "okay", "now", "just"]

CUSTOM_STOPWORDS = FILLER_VOCAB + GENERIC_STOPWORDS

ENTITY_CLASSES = ["action", "target", "correction_connector"]

DATASET_PATH_SIMPLE = "chatette/complex/*.json"
DATASET_PATH_COMPLEX = "chatette/simple/*.json"


# method to load vocab exported from ExportVocab.py
def _load_vocab(path: str) -> dict:
    with open(path, "r") as f:
        vocab = json.load(f)

    for class_name in ENTITY_CLASSES:
        if class_name not in vocab:
            raise KeyError(f"vocab.json is missing required class named {class_name}")

    return vocab


VOCAB = _load_vocab(VOCAB_PATH)

_VOCAB_COMPILED = {
    class_name: [
        (entry, re.compile(r"(?<!\w)" + re.escape(entry.lower()) + r"(?!\w)"))
        for entry in sorted(entries, key=len, reverse=True)
    ]
    for class_name, entries in VOCAB.items()
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


def load_dataset(path_pattern: str, complexity_label: str) -> list[Example]:
    import glob

    files_found = sorted(glob.glob(path_pattern))
    if not files_found:
        raise FileNotFoundError(f"No files matched {path_pattern}")

    examples = []
    for path in files_found:
        with open(path, "r") as f:
            raw = json.load(f)
        for example in raw["rasa_nlu_data"]["common_examples"]:
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
def run_gazetteer_on_example(sentence: str) -> tuple[dict, float]:
    start = time.perf_counter()

    sentence_l = sentence.lower()
    predicted = {class_name: None for class_name in ENTITY_CLASSES}
    # tracking the start index of the current best (rightmost) match per class
    best_start = {class_name: -1 for class_name in ENTITY_CLASSES}

    for class_name, compiled_entries in _VOCAB_COMPILED.items():
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
    print("Loading datasets...")
    simple_examples = load_dataset(DATASET_PATH_SIMPLE, "simple")
    complex_examples = load_dataset(DATASET_PATH_COMPLEX, "complex")
    examples = simple_examples + complex_examples
    print(
        f"Loaded {len(simple_examples)} simple / {len(complex_examples)} complex "
        f"({len(examples)} total)"
    )

    print("Running pure gazetteer matching...")
    predictions = []
    for ex in examples:
        pred, _ = run_gazetteer_on_example(ex.sentence)
        predictions.append(pred)

    print("Computing metrics...")
    metrics = evaluate(examples, predictions)

    print("Getting latency for 5 passes...")
    all_latencies = []
    for _ in range(5):
        for ex in examples:
            _, ms = run_gazetteer_on_example(ex.sentence)
            all_latencies.append(ms)
    latency = {
        "mean_ms": statistics.mean(all_latencies),
        "median_ms": statistics.median(all_latencies),
        "stdev_ms": statistics.stdev(all_latencies),
        "n_samples": len(all_latencies),
    }

    output = {"metrics": metrics, "latency": latency}
    with open("gazetteer_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved full results to gazetteer_results.json")


if __name__ == "__main__":
    main()
