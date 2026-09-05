# Transformers vs Rule-based Extraction for Multi-word NER: Accuracy and Latency Trade-offs in VUI Pipelines

We investigate the benefits of transformer-based keyword extraction over traditional statistical methods in Voice-Controlled User Interfaces (VUI).

Chatette, the Rasa NLU dataset generator used for evaluating the approaches in the paper - https://github.com/SimGus/Chatette

Speak Up, the implementation of our GAMEBERT on PyPi - https://pypi.org/project/voice-speak-up/

GAMEBERT model on HuggingFace - https://huggingface.co/Saggarwal/GAMEBERT

## Baseline evaluation

Install the Python dependencies with `pip install -r requirements.txt`, then run
all four baseline evaluations with:

```bash
python Reproduce.py
```

The individual entry points are in `src/`:

```bash
python src/EvaluateRAKEClosed.py
python src/EvaluateRAKEOpen.py
python src/EvaluateGazetteerClosed.py
python src/EvaluateGazetteerOpen.py
```

Closed-vocabulary data is loaded from `data/splits/closed`. Open-vocabulary
evaluations build their lookup vocabulary from `data/splits/open/vocab_train`
and evaluate against `data/splits/open/vocab_test`.