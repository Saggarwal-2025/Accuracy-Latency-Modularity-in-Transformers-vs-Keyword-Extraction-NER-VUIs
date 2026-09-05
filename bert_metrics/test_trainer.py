# purpose of class is to evaluate GAMEBERT (token-classification / BIO) architecture on
# correctly tagging actions, targets and corrections in different complexities of sentences
# with possible filler/stop/negation words included.
#
# Unlike RAKE, GAMEBERT is a tagger predicting directly over sentence tokens, so predicted
# spans are scored straight against entity["value"] -- no closed-vocab canonicalization step
# (the dataset's start/end offsets confirm value == raw span text, so there's no
# surface-form-vs-canonical-form gap to bridge here).
#
# Correctness (accuracy/precision/recall/f1) is reported as single point-estimates per
# entity per bucket, same as RAKE. Latency is reported as mean/median/SD over 5 passes,
# same as RAKE's benchmark_latency.

import json
import time
import statistics
from dataclasses import dataclass
import glob

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MODEL_NAME = "Saggarwal/GAMEBERT"
FILLER_VOCAB = ["um", "uh", "eh"]

ENTITY_CLASSES = ["action", "target", "correction_connector"]

# maps GAMEBERT's BIO tag type (the part after "B-"/"I-", upper-cased) to the
# ENTITY_CLASSES naming used by the dataset. Check model.config.id2label
# and adjust this if GAMEBERT's tag names differ from what's assumed here.
TAG_TYPE_TO_CLASS = {
    "ACTION": "action",
    "TARGET": "target",
    "CORRECTION": "correction_connector",
}

DATASET_PATH_SIMPLE = "../other_methodology/chatette/closed_vocab/simple/*.json"
DATASET_PATH_COMPLEX = "../other_methodology/chatette/closed_vocab/complex/*.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# class defining example to be evaluated on, storing the sentence, actual labels found,
# complexity and if the sentence has filler -- identical to RAKE's Example
@dataclass
class Example:
    sentence: str
    actual_labels: dict  # {"action": "jump", "target": "left", "correction_connector": None}

    complexity: str = "simple"  # "simple" or "complex" - if correction or negation

    has_filler: bool = False


# converting chatette dataset's entity list into only one value per class -- identical to RAKE
def _entities_to_actual_labels(entities: list) -> dict:
    labels = {class_name: None for class_name in ENTITY_CLASSES}

    for entity in entities:
        entity_class = entity["entity"]

        if entity_class in labels:
            labels[entity_class] = entity["value"]

    return labels


# method to load chatette's generated rasa nlu json output directly for each complexity
# type separately -- identical to RAKE's loader, so both scripts see the exact same examples
def load_dataset(paths, complexity_label: str) -> list[Example]:
    if isinstance(paths, str):
        file_list = sorted(glob.glob(paths))
    else:
        file_list = paths

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


# method to load GAMEBERT + tokenizer once
def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# method to decode a token-classification model's BIO predictions on one sentence into
# {entity_class: raw_span_text}, taking the LAST span of each type -- matches
# _entities_to_actual_labels' ground-truth semantics (post-correction value wins).
#
# Decodes at the WORD level, not the raw token level: only the first subword's tag
# per word is used, and the character span is extended to cover every subword in
# that word. This matters because WordPiece splits unseen words (e.g. "Goomba" ->
# "goo" + "##mba") and standard NER fine-tuning only trains a real label on the
# first subword of each word (trailing "##" subwords get -100/ignored during
# training), so their predicted tags are untrained noise and must not be allowed
# to truncate the span.
def decode_bio_spans(sentence: str, tokenizer, id2label, logits, offset_mapping, word_ids) -> dict:
    pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    tags = [id2label[i] for i in pred_ids]

    # collapse subword tokens into words: first-subword's tag is authoritative,
    # word's char span extends across all its subwords
    word_order = []
    word_tag = {}
    word_span = {}

    for tag, (start, end), w_id in zip(tags, offset_mapping, word_ids):
        if w_id is None:  # special token (CLS/SEP/PAD)
            continue
        if w_id not in word_tag:
            word_tag[w_id] = tag
            word_span[w_id] = [start, end]
            word_order.append(w_id)
        else:
            word_span[w_id][1] = end  # extend to cover this trailing subword

    spans = {}  # class_name -> raw text, LAST occurrence wins
    current_type = None
    current_start = None
    current_end = None

    def flush():
        nonlocal current_type, current_start, current_end
        if current_type is not None:
            class_name = TAG_TYPE_TO_CLASS.get(current_type)
            if class_name:
                # overwrite (keep LAST occurrence per class) to match
                # _entities_to_actual_labels' ground-truth semantics, where a
                # later-in-sentence span (e.g. the post-correction action) is
                # the intended label, not the first one mentioned
                spans[class_name] = sentence[current_start:current_end]
        current_type = None
        current_start = None
        current_end = None

    for w_id in word_order:
        tag = word_tag[w_id]
        start, end = word_span[w_id]

        if tag == "O":
            flush()
            continue

        prefix, _, tag_type = tag.partition("-")
        tag_type = tag_type.upper()

        if prefix == "B":
            flush()
            current_type = tag_type
            current_start = start
            current_end = end
        elif prefix == "I" and current_type == tag_type:
            current_end = end
        else:
            # I- tag with no matching B- (or mismatched type) -- treat as new span start
            flush()
            current_type = tag_type
            current_start = start
            current_end = end

    flush()
    return spans


# method to test GAMEBERT on an example -- mirrors RAKE's run_rake_on_example signature.
# Predicted span text is scored directly against entity["value"] (normalized), no vocab step.
def run_bert_on_example(model, tokenizer, sentence: str) -> tuple[dict, float]:
    start = time.perf_counter()

    enc = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
    )
    offset_mapping = enc.pop("offset_mapping").squeeze(0).tolist()
    word_ids = enc.word_ids(batch_index=0)  # requires a fast (Rust-backed) tokenizer
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    id2label = model.config.id2label
    raw_spans = decode_bio_spans(sentence, tokenizer, id2label, logits, offset_mapping, word_ids)

    predicted = {class_name: None for class_name in ENTITY_CLASSES}
    for class_name, raw_text in raw_spans.items():
        predicted[class_name] = raw_text.strip().lower()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return predicted, elapsed_ms


# method to compute accuracy/precision/recall/f1-score for one entity class -- identical to RAKE
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# method to aggregate metrics for every entity by complexity / filler-robustness -- identical to RAKE
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
            gt_raw = ex.actual_labels.get(entity)
            gt = gt_raw.strip().lower() if gt_raw is not None else "none"
            pd_raw = pred.get(entity)
            pd_ = pd_raw if pd_raw is not None else "none"
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


# method to run 5 passes over test set to get latency values -- mirrors RAKE's benchmark_latency
def benchmark_latency(examples: list[Example], model, tokenizer, n_passes: int = 5) -> dict:
    all_latencies = []

    for _ in range(n_passes):
        for example in examples:
            _, ms = run_bert_on_example(model, tokenizer, example.sentence)
            all_latencies.append(ms)

    return {
        "mean_ms": statistics.mean(all_latencies),
        "median_ms": statistics.median(all_latencies),
        "stdev_ms": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0,
        "n_samples": len(all_latencies),
    }


# main to run everything together
def main():
    print("Loading datasets...")

    simple_examples = load_dataset(DATASET_PATH_SIMPLE, "simple")
    complex_examples = load_dataset(DATASET_PATH_COMPLEX, "complex")
    examples = simple_examples + complex_examples

    print(
        f"Loaded {len(simple_examples)} simple / {len(complex_examples)} complex "
        f"({len(examples)} total)"
    )

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = build_model()

    print("Running GAMEBERT inference + BIO decoding...")

    predictions = []

    for example in examples:
        pred, _ = run_bert_on_example(model, tokenizer, example.sentence)
        predictions.append(pred)

    print("Computing metrics...")

    metrics = evaluate(examples, predictions)

    print("Getting latency for each of 5 passes...")
    latency = benchmark_latency(examples, model, tokenizer, n_passes=5)

    output = {"model": MODEL_NAME, "metrics": metrics, "latency": latency}

    with open("bert_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved full results to bert_results.json")


if __name__ == "__main__":
    main()