# Quora Question Pairs — Semantic Similarity

Predicting whether two Quora questions ask the same thing, using the public
[Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs)
dataset. Built for a university machine learning course competition, where this
submission placed **3rd**.

The task is deceptively hard. *"How do I learn Python?"* and *"What is the best
way to learn Python?"* are duplicates; *"How do I learn Python?"* and *"How do I
learn Python 2?"* are not. Lexical overlap is high in both cases, so surface
similarity alone gets you very little.

## Results

| Metric | Value |
|---|---|
| Accuracy | **0.883** |
| Validation logloss | **0.250267** |
| Training logloss | 0.222067 |
| Best iteration | 880 of 1000 |

Training and validation logloss tracking closely — 0.222 against 0.250 — is the
part worth noting: the gap stayed narrow through 880 rounds, so the model was
still generalising rather than memorising when early stopping picked the
iteration.

## How it works

Nearly all the gain came from feature engineering, not model selection. The
classifier was a stock gradient-booster; what moved the score was giving it
features that capture *semantic* rather than *lexical* similarity:

- **TF-IDF with truncated SVD** — vectorise both questions, then reduce to
  dense latent components. SVD rather than raw TF-IDF because the sparse
  high-dimensional form buries the signal a tree-based model can actually split
  on.
- **Fuzzy string distances** — token sort and token set ratios, which are
  robust to word reordering in a way that exact matching is not. *"Best way to
  learn Python"* and *"Python, what is the best way to learn"* score high here
  and near zero on naive overlap.
- **Cosine distance** between vector representations, as a direct
  angular-similarity signal.
- **Porter stemming** and NLTK preprocessing, so inflectional variants collapse
  before any of the above is computed.

Those features then fed an XGBoost classifier tuned by early stopping on the
validation split.

The broader lesson is the ordinary one in applied ML: a better model on weak
features loses to a stock model on good ones, and the time is better spent on
the representation.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook "Quora Question Pairs.ipynb"
```

The dataset is not committed — download `train.csv` from the
[competition page](https://www.kaggle.com/competitions/quora-question-pairs/data)
and point the loader at it.

## Stack

Python, XGBoost, scikit-learn, NLTK, fuzzywuzzy, pandas, NumPy, SciPy,
matplotlib, seaborn.
