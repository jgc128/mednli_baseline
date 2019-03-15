MedNLI Baseline
===============
A simple baseline for Natural Language Inference in clinical domain using the MedNLI dataset.
Includes simplified CBOW and InferSent models from the corresponding paper. 

Installation
------------

1. Clone this repo: `git clone https://github.com/jgc128/mednli_baseline.git`
1. Install NumPy: `pip install numpy==1.15.2`
1. Install PyTorch v0.4.1: `pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl` (see https://pytorch.org/ for details)
1. Install requirements: `pip install -r requirements.txt`

Downloading the dataset, word embeddings, and pre-trained models
----------------------------------------------------------------
1. Create the `./data` directory inside the cloned repository
    1. Create the `./data/cache` directory 
1. Download MedNLI: https://jgc128.github.io/mednli/
    1. Extract the content of the `mednli_data.zip` archive into the `./data/mednli` dir (`unzip -d data/mednli mednli_data.zip`)
1. Download word embeddings (see the table below) and put the `*.pickled` files into the `./data/word_embeddings/` dir (`wget -P data/word_embeddings/ https://mednli.blob.core.windows.net/shared/word_embeddings/https://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled`)
1. Download pre-trained models (see below) and put the `*.pkl` and the `*.pt` files into the `./data/models/` dir

### Word embeddings
| Word Embedding  | Link |
| ------------- | ------------- |
|glove |  [glove.840B.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove.840B.300d.pickled) |
|mimic |  [mimic.fastText.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/mimic.fastText.no_clean.300d.pickled) |
|bio_asq | [bio_asq.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/bio_asq.no_clean.300d.pickled) |
|wiki_en | [wiki_en.fastText.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/wiki_en.fastText.300d.pickled) |
|wiki_en_mimic |  [wiki_en_mimic.fastText.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/wiki_en_mimic.fastText.no_clean.300d.pickled) |
|glove_bio_asq |  [glove_bio_asq.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove_bio_asq.no_clean.300d.pickled) |
|glove_bio_asq_mimic |[glove_bio_asq_mimic.no_clean.300d.pickled](https://mednli.blob.core.windows.net/shared/word_embeddings/glove_bio_asq_mimic.no_clean.300d.pickled) |

### Models

| Model     | Embeddings          | MedNLI Dev accuracy | Files |
|-----------|---------------------|---------------------|-------|
| CBOW      | mimic               | 0.670               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.simple.mimic.128.1tnliqel.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.simple.mimic.128.1tnliqel.pt) |
| InferSent | glove               | 0.743               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.glove.128.fscwzdnn.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.glove.128.fscwzdnn.pt) |
| InferSent | mimic               | 0.783               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.mimic.128.sariedpg.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.mimic.128.sariedpg.pt) |
| InferSent | wiki_en             | 0.763               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.wikien.128.2k7kname.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.wikien.128.2k7kname.pt) |
| InferSent | wiki_en_mimic       | 0.774               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.wikienmimic.128.hcehjp7m.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.wikienmimic.128.hcehjp7m.pt) |
| InferSent | glove_bio_asq_mimic | 0.770               | [model spec](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.glovebioasqmimic.128.n8d0l13c.pkl) / [model weights](https://mednli.blob.core.windows.net/shared/mednli_baseline/mednli.infersent.glovebioasqmimic.128.n8d0l13c.pt) |



Using a pre-training model
--------------------------
Run the `predict.py` file with three arguments:
1. Path to the model specification file (`*.pkl`)
1. Input file in the `jsonl` format (see `mli_dev_v1.jsonl`) or the `\t`-separated premise and hypothesis (see [test_input.txt](https://mednli.blob.core.windows.net/shared/test_input.txt)) 
1. Output file `.csv` to save predicted probabilities of each of the three classes (contradiction, entailment, and neutral)

Notes:
1. The model weights file (`*.pt`) should be located in the same dir as the model specification file (`*.pkl`)
1. In case of the `jsonl` format the sentences are taken from the `sentence1_binary_parse` and `sentence2_binary_parse` fields,
 where the `sentence1` is the premise and `sentence2` is the hypothesis. All other fields are optional

Example command to run the prediction:
```
python predict.py data/models/mednli.infersent.mimic.128.saek2t5q.pkl data/input_test.txt data/predictions_test.csv
```

Training the model
------------------

Run the `train.py` file. The options are set in the `config.py` file. Command-line interface is coming soon!
By default, the model specification and the model weights are saved in the `./data/models` dir.

Training the feature based system
------------------

To run a traditional feature based system, run the `train_feature_based.py` file. 
This system achieves 0.523 accuracy on the dev set using a gradient boosting classifier 
with features based on word overlaps, tf-idf similarities, word embeddings similarities, and blue scores.


# Reference

Romanov, A., & Shivade, C. (2018). Lessons from Natural Language Inference in the Clinical Domain. arXiv preprint arXiv:1808.06752.  
https://arxiv.org/abs/1808.06752


```
@article{romanov2018lessons,
	title = {Lessons from Natural Language Inference in the Clinical Domain},
	url = {http://arxiv.org/abs/1808.06752},
	abstract = {State of the art models using deep neural networks have become very good in learning an accurate mapping from inputs to outputs. However, they still lack generalization capabilities in conditions that differ from the ones encountered during training. This is even more challenging in specialized, and knowledge intensive domains, where training data is limited. To address this gap, we introduce {MedNLI} - a dataset annotated by doctors, performing a natural language inference task ({NLI}), grounded in the medical history of patients. We present strategies to: 1) leverage transfer learning using datasets from the open domain, (e.g. {SNLI}) and 2) incorporate domain knowledge from external data and lexical sources (e.g. medical terminologies). Our results demonstrate performance gains using both strategies.},
	journaltitle = {{arXiv}:1808.06752 [cs]},
	author = {Romanov, Alexey and Shivade, Chaitanya},
	urldate = {2018-08-27},
	date = {2018-08-21},
	eprinttype = {arxiv},
	eprint = {1808.06752},
}
