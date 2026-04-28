# Multilingual NLP Analysis of Movie Reviews — README

This repository contains a mid-term project that analyses movie-review
sentiment across three languages (Korean, Chinese, English) through a
uniform NLP pipeline: tokenization, embeddings, attention, and
classification with Base / SFT / LoRA variants of
distilbert-base-multilingual-cased.

## 1. Data sources
## Dataset Download

Due to GitHub file size limitations, the processed datasets are hosted on Google Drive:

 https://drive.google.com/drive/folders/1c5Npfvftr8Nq1Hzv69k6QWt4OOE8C4YU



| Language | Corpus | URL |
|---|---|---|
| Korean  | NSMC                | https://huggingface.co/datasets/e9t/nsmc |
| Chinese | Douban Movie Short Comments | https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments |
| English | IMDB 50K Movie Reviews | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews |

After cleaning, 2,000 documents per language are retained (1,000 positive
+ 1,000 negative), for a total of 6,000 samples. The Chinese sample
covers 28 distinct movies through stratified sampling; Korean and English
are drawn from the full pool of their respective corpora.

Output file: `data.csv` with fields `text, label, source, language, movie`.

## 2. Directory layout expected by the notebook
After downloading, please place the files as follows:

project/
├── 1.ipynb                      # the pipeline notebook (this repository)
├── nsmc/
│   └── nsmc_train.csv
├── douban/
│   └── DMSC.csv
├── IMDB/
│   └── IMDB Dataset.csv
└── img/                         # figures written here by the notebook
```


## 3. Python environment

Tested on Python 3.10 with the following packages (CUDA optional):

```
numpy          pandas          matplotlib      seaborn
scikit-learn   gensim          jieba           kiwipiepy
torch          transformers    peft            jupyter
```

Install with:

```
pip install numpy pandas matplotlib seaborn scikit-learn gensim \
            jieba kiwipiepy torch transformers peft jupyter
```

The notebook downloads `distilbert-base-multilingual-cased` automatically
the first time it runs.

## 4. How to run

From the project root, launch Jupyter and open the notebook:

```
jupyter notebook 1.ipynb
```

Then run all cells (Cell → Run All). The notebook will:

1. Load NSMC, Douban, and IMDB from the paths listed above.
2. Clean, deduplicate, stratify-sample, and save `data.csv`.
3. Compute tokenization statistics (Kiwi / Jieba / whitespace vs. BPE).
4. Train Word2Vec vectors and extract BERT embeddings.
5. Render attention heatmaps for contrastive and non-contrastive
   sentences in each language.
6. Fine-tune three classifier variants (Base, SFT, LoRA) and evaluate.
7. Produce error analysis and multilingual summary.

All eleven figures are written to `./img/*.png`.

## 5. Reproducibility

A single random seed (`SEED = 42`) controls all NumPy, PyTorch, Python,
`random`, `transformers.set_seed`, and `sklearn` splits.
`torch.backends.cudnn.deterministic` is set to True when CUDA is used.

## 6. Hardware used

The pipeline was run on a single NVIDIA GPU. Total wall-clock time on an
RTX-class card is roughly 10–15 minutes: most of that is the SFT pass,
which updates all 135 M parameters for three epochs.

## 7. Known limitations

See Section 10 of the report for a full discussion. Key points:

- Sample size is 2,000 per language; Word2Vec is undertrained at this
  scale.
- Multi-movie stratification is only enforced for Chinese because NSMC
  and IMDB do not expose a movie-id field.
- Attention analysis uses two sentences per language rather than a
  corpus-level aggregate.
- The LoRA configuration (rank 8, on `q_lin` and `v_lin` only) is one
  point in the adapter design space; it was not systematically tuned.
