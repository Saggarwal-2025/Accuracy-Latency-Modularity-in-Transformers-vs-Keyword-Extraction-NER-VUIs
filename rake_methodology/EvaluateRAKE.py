# purpose of class is to evaluate RAKE architecture on correctly tagging actions, targets and corrections in different complexities of sentences with possible filler/stop/negation words included

import json
import re
import time
import statistics
from dataclasses import dataclass

from rake_nltk import Rake
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

DATASET_PATH = "output-2.json"
VOCAB_PATH = "vocab.json"
FILLER_VOCAB = ["um", "uh", "eh"]
GENERIC_STOPWORDS = ["please", "can", "you", "okay", "now", "just"]

CUSTOM_STOPWORDS = FILLER_VOCAB + GENERIC_STOPWORDS

ENTITY_CLASSES = ["action", "target", "correction_connector"]
NEGATION_MARKERS = {"no", "wait", "don't", "dont", "actually", "instead"}


# method to load vocab exported from ExportVocab.py
def _load_vocab(path: str) -> dict:
    try:
        with open(path, "r") as f:

            vocab = json.load(f)

        for class_name in ENTITY_CLASSES:
            if class_name not in vocab:
                raise KeyError(
                    f"vocab.json is missing required class named {class_name}"
                )

        return vocab

    except FileNotFoundError:
        print(
            f"WARNING: {path} not found -- using placeholder vocab. "
            f"Results will NOT be meaningful until you export your real "
            f"~[action]/~[target]/~[correction_connector] slot lists."
        )
        return {
            "action": ["jump", "run", "crouch", "shoot"],
            "target": ["left", "right", "up", "down"],
            "correction_connector": ["no", "wait", "actually", "scratch that"],
        }


VOCAB = _load_vocab(VOCAB_PATH)
_VOCAB_COMPILED = {
    cls: [
        (entry, re.compile(r"(?<!\w)" + re.escape(entry.lower()) + r"(?!\w)"))
        for entry in sorted(entries, key=len, reverse=True)
    ]
    for cls, entries in VOCAB.items()
}


# class defining example to be valuated on, storing the sentence, actual labels found, complexity and if the sentence has filler
@dataclass
class Example:
    sentence: str
    actual_labels: dict  # {"action": "jump", "target": "left", "correction": None}

    complexity: str = "simple"  # "simple" or "complex" - if correction or negation

    has_filler: bool = False


# method to derive complexity from test sentences
def _infer_complexity(entities: list, text: str) -> str:
    entity_types = {e["entity"] for e in entities}
    lowered = text.lower()

    if "correction_connector" in entity_types:
        return "complex"

    if any(marker in lowered.split() for marker in NEGATION_MARKERS):
        return "complex"

    return "simple"


# converting chatette dataset's entity list into only one value per class
def _entities_to_actual_labels(entities: list) -> dict:
    labels = {class_name: None for class_name in ENTITY_CLASSES}

    # for every entity,
    for entity in entities:
        entity_class = entity["entity"]  # getting class belonging to the entity

        # if same class has already come before/inferring is before the correction token as sentence can only have at most one of each class of entity
        if entity_class in labels:
            labels[entity_class] = entity[
                "value"
            ]  # overwriting to new intended value for class entity

    return labels


# method to load chatette's generated rasa nlu json output directly
def load_dataset(path: str) -> list[Example]:
    with open(path, "r") as f:
        raw = json.load(f)

    common_examples = raw["rasa_nlu_data"]["common_examples"]

    examples = []

    for example in common_examples:
        text = example["text"]  # getting actual text value
        entities = example.get("entities", [])  # getting tagged entities

        examples.append(
            Example(
                sentence=text,
                actual_labels=_entities_to_actual_labels(entities),
                complexity=_infer_complexity(entities, text),
                has_filler=any(w in text.lower().split() for w in FILLER_VOCAB),
            )
        )

    return examples


# method to initialize RAKE extractor based on stopwords to look out for
def build_rake() -> Rake:
    return Rake(stopwords=set(CUSTOM_STOPWORDS) | set(Rake().stopwords))


# method to make RAKE's ranked phrases to action/target/correction labels using slots in chatette vocab, checking each phrase for a known vocab entry with the set being closed and avoiding the risk of adding semantic understanding RAKE's architecture doesn't actually allow
def map_phrases_to_entities(ranked_phrases: list[str]) -> dict:
    predicted = {cls: None for cls in ENTITY_CLASSES}

    for phrase in ranked_phrases:
        phrase_l = phrase.lower()

        for class_name in ENTITY_CLASSES:
            if predicted[class_name] is not None:
                continue
            for entry, pattern in _VOCAB_COMPILED[class_name]:
                if pattern.search(phrase_l):
                    predicted[class_name] = entry
                    break

    return predicted


# method to test rake on an example
def run_rake_on_example(rake: Rake, sentence: str) -> tuple[dict, float]:
    start = time.perf_counter()  # getting start tim

    rake.extract_keywords_from_text(sentence)
    ranked_phrases = rake.get_ranked_phrases()
    predicted = map_phrases_to_entities(ranked_phrases)

    elapsed_ms = (time.perf_counter() - start) * 1000  # calculating ms latency

    return predicted, elapsed_ms


# method to compute accuracy/precision/recall/f1-score for one entity class
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # returning dictionary with mapping
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# method to aggregate metrics for every entity by complexity on filler-robustness/sentences containing filler words
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


# method to run 5 passes over test set to get latency values
def benchmark_latency(examples: list[Example], rake: Rake, n_passes: int = 5) -> dict:
    all_latencies = []

    # for each of the 5 passes
    for _ in range(n_passes):

        # for every test sentence/example,
        for example in examples:
            _, ms = run_rake_on_example(
                rake, example.sentence
            )  # returning latency values from method defined above

            all_latencies.append(ms)

    return {
        "mean_ms": statistics.mean(all_latencies),
        "median_ms": statistics.median(all_latencies),
        "stdev_ms": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0,
        "n_samples": len(all_latencies),
    }


# main to run everything together
def main():
    print("Loading dataset...")
    examples = load_dataset(DATASET_PATH)

    print(f"Loaded {len(examples)} test examples.")

    rake = build_rake()

    print("Running RAKE extraction + entity mapping...")

    predictions = []

    for example in examples:
        pred, _ = run_rake_on_example(rake, example.sentence)
        predictions.append(pred)

    print("Computing metrics...")

    metrics = evaluate(examples, predictions)

    print("Getting latency for each of 5 passes...")
    latency = benchmark_latency(examples, rake, n_passes=5)

    # saving all results to access later
    output = {"metrics": metrics, "latency": latency}

    with open("rake_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved full results to rake_results.json")


if __name__ == "__main__":
    main()
