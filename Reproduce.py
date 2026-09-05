# this class acts as a one-command reproduction of this paper's headline result of Table 1, running the RAKE and gazetteer baseline evaluations for closed and open datasets, and prints a summary of the generated result files

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
SRC = REPO_ROOT / "src"


def run_step(description: str, cmd: list[str], cwd: Path, required: bool = True):
    print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:

        msg = f"Step failed: {description}"

        if required:
            print(f"ERROR: {msg}. Aborting.")
            sys.exit(1)

        else:
            print(
                f"WARNING: {msg}. Continuing without this component "
                f"(see README for how to run it manually once dependencies "
                f"are available)."
            )

    return result.returncode == 0


def print_summary():
    print(f"\n{'=' * 60}\nHEADLINE RESULT SUMMARY (Table 1 - Overall)\n{'=' * 60}")

    for name, path in [
        ("RAKE closed", REPO_ROOT / "rake_closed_results.json"),
        ("RAKE open", REPO_ROOT / "rake_open_results.json"),
        ("Gazetteer closed", REPO_ROOT / "gazetteer_closed_results.json"),
        ("Gazetteer open", REPO_ROOT / "gazetteer_open_results.json"),
        ("GAMEBERT closed", REPO_ROOT / "gamebert_closed_results.json"),
        ("GAMEBERT open", REPO_ROOT / "gamebert_open_results.json"),
    ]:

        if not path.exists():
            print(f"{name}: [not run -- see above]")
            continue
        with open(path) as f:
            data = json.load(f)
        overall = data["metrics"]["overall"]
        print(f"\n{name}:")
        for entity, vals in overall.items():
            print(f"  {entity:22s} acc={vals['accuracy']:.3f}  f1={vals['f1']:.3f}")


def main():
    for label, script, dataset in [
        ("closed RAKE", "rake/EvaluateRAKEClosed.py"),
        ("open RAKE", "rake/EvaluateRAKEOpen.py"),
        ("closed gazetteer", "gazetteer/EvaluateGazetteerClosed.py"),
        ("open gazetteer", "gazetteer/EvaluateGazetteerOpen.py"),
        ("closed GAMEBERT", "EvaluateGAMEBERT.py", "closed"),
        ("open GAMEBERT", "EvaluateGAMEBERT.py", "open"),
    ]:

        command = [sys.executable, script]

        # letting you know what dataset is currently being evaluated on
        if dataset:
            command += ["--dataset", dataset]

        run_step(f"Running {label} evaluation", command, cwd=SRC)

    print_summary()


if __name__ == "__main__":
    main()
