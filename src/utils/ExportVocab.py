# purpose of class is to scan every entity value that appears in the generated chatette dataset and creates out list for every entity class with no duplicates for EvaluateRAKE.py to take in

import json
import glob
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "splits"

ENTITY_CLASSES = ["action", "target", "correction_connector"]


# method to load examples currently existing in chatette output json file
def load_common_examples(chatette_files: list[str]) -> list[dict]:
    files_found = []
    examples = []

    # for every pattern in chatette files passed in,
    for pattern in chatette_files:
        # read and add each matching file
        files_found.extend(glob.glob(pattern))

    # loading the entire contents of the file from each of the found files
    for path in files_found:
        with open(path, "r") as f:
            raw = json.load(f)

        # only adding to examples list only the actual examples, not the set up parts of the json
        examples.extend(raw["rasa_nlu_data"]["common_examples"])

    print(
        f"Loaded {len(examples)} examples from {len(files_found)} file(s): "
        f"{files_found}"
    )

    # returning all examples
    return examples


# creates three sets containing every single unique value currently included for each of the classes in the examples loaded in from the chatette output json file (action, target, correction)
def build_vocab(examples: list[dict]) -> dict:
    vocab = {class_name: set() for class_name in ENTITY_CLASSES}

    for example in examples:
        for entity in example.get("entities", []):
            class_name = entity["entity"]

            if class_name in vocab:
                vocab[class_name].add(entity["value"])

    # sorting values for readability and reproducibility of methods
    return {class_name: sorted(values) for class_name, values in vocab.items()}


# main method putting everything together
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("closed", "open"), default="closed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.dataset == "closed":
        root = DATA_ROOT / "closed"
        chatette_path = [
            str(root / "simple" / "*.json"),
            str(root / "complex" / "*.json"),
        ]
    else:
        root = DATA_ROOT / "open" / "vocab_train"
        chatette_path = [
            str(root / "simple" / "train" / "*.json"),
            str(root / "complex" / "train" / "*.json"),
        ]

    examples = load_common_examples(chatette_path)
    vocab = build_vocab(examples)

    for class_name, values in vocab.items():
        print(f"{class_name}: {len(values)} unique values")

    output_path = args.output or REPO_ROOT / f"vocab_{args.dataset}.json"
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved vocab to {output_path}")


if __name__ == "__main__":
    main()
