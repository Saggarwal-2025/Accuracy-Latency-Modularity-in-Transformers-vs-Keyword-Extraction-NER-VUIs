import sys

from paper.GAMEBERT_data.src.rake.EvaluateRAKE import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--dataset", "open"]
    main()
