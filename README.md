# Interlyst
Intersection Flow Optimization Model

Quick start:
```
# 1. Install dependacies
pip install -r requirements.txt

# 2. Generate sample data for training
python -m data.generate --n 20000 --workers 8 --out data/dataset.csv

# 2a. Simple model training
python -m model.sklearn_baseline --csv data/dataset.csv

# 2b. Or more capable model training
python -m model.train --csv data/dataset.csv --epochs 150
python -m eval.compare --csv data/dataset.csv --ckpt model/checkpoint.pt
```

You can launch visual editor with or without training the model, but expect worse results in optimizer:
```
python -m gui.app
```