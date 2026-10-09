<div align="center">

# 🧠 Multiclass Sentiment Classification with BERT

**Fine-tune `bert-base-cased` with TensorFlow to classify text into five sentiment classes.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## ✨ Overview

`bert_multiclass_nlp.ipynb` fine-tunes BERT on a five-level sentiment dataset (`data/train.tsv`):

| Label | Class |
|---|---|
| 0 | 😡 Negative |
| 1 | 🙁 A little negative |
| 2 | 😐 Neutral |
| 3 | 🙂 A little good |
| 4 | 😄 Good |

**Pipeline**

1. Load the TSV data and one-hot **labelize** the five classes.
2. Tokenize with `BertTokenizer` (`bert-base-cased`, sequence length 256) and build `input_ids` + `attention_mask` tensors.
3. Put a Keras classification head on top of `TFBertModel`.
4. Train and evaluate - about **67 % validation accuracy after one epoch** in the recorded run (more epochs / a larger GPU budget would improve it).

## 🚀 Getting Started

The notebook was run on Google Colab (GPU). Locally:

```bash
git clone https://github.com/Arashomranpour/multiclass_bert.git
cd multiclass_bert
pip install -r requirements.txt
jupyter notebook bert_multiclass_nlp.ipynb
```

Place the dataset (a Rotten-Tomatoes-style `train.tsv` with phrases and 0-4 sentiments) in `data/`.

## 📁 Project Structure

```
.
├── bert_multiclass_nlp.ipynb   # Data prep, BERT fine-tuning, evaluation
└── requirements.txt
```

## 🛠️ Tech Stack

`TensorFlow` · `Hugging Face Transformers (BERT)` · `pandas` · `NumPy` · `tqdm`
