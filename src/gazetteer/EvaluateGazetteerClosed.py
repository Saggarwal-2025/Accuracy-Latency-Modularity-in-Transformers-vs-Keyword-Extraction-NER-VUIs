import sys

from paper.GAMEBERT_data.src.gazetteer.EvaluateGazetteer import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--dataset", "closed"]
    main()
