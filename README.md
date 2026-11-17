# Interlyst
Intersection Flow Optimization Model

Quick start:
```
# 1. Install dependacies
pip install -r requirements.txt

# 2. Generate sample data for training
python -m data.generate --n 20000 --workers 8 --out output/dataset.csv

# 3a. Simple model training
python -m model.sklearn_baseline --csv output/dataset.csv

# 3b. Or more capable model training
python -m model.train --csv output/dataset.csv --epochs 150
python -m eval.compare --csv output/dataset.csv --ckpt output/checkpoint.pt
```

You can launch visual editor with or without training the model, but expect worse results in optimizer:
```
python -m interface.app
```