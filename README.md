# Knowledge-Enabled Textual-Entailment

Natural Language Inference is fundamental to many Natural Language Processing applications such as semantic search and question answering. The task of NLI has gained significant attention in the recent times due to the release of fairly large scale, challenging datasets. Present approaches that address NLI are largely focused on learning based on the given text in order to classify whether the given premise entails, contradicts, or is neutral to the given hypothesis. On the other hand, techniques for Inference, as a central topic in artificial intelligence, has had knowledge bases playing an important role, in particular for formal reasoning tasks. While, there are many open knowledge bases that comprise of various types of information, their use for natural language inference has not been well explored. In this work, we present a simple technique that can harnesses knowledge bases, provided in the form of a graph, for natural language inference.

## Setup

Before running the experiment for the first time, you will need to
download and preprocess the datasets and the word embedding files.

To run the setup scripts, clone the repository and enter the `./scripts`

```bash
git clone <to be replace by actual git url>
cd repo_root/entailment/scripts
```

### Download Datasets and Embeddings

The following script will download and unzip the NLI datasets (SciTail, SNLI, MultiNLI) into `./data/nli_datasets`

```bash
bash ./download_datasets.sh
```

And the following script will download and unzip the word embeddings (GloVe, ConceptNet-PPMI) into `./data/glove` and `./data/conceptnet`, respectively.

```bash
bash ./download_embeddings.sh
```

### Install Required Packages

Before installing the required packages, it is recommended that you
create a virtual environment dedicated for each of the project,
especially if you are using different versions of the packages. Some of
the popular options include [`conda`](https://conda.io/miniconda.html),
[`pipenv`](https://github.com/pypa/pipenv), or the naive
 [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/).
 Most of the code are written in Python 3.6, but should be compatible
 with other versions of Python.

 Going back to the root directory of the project if you are still in
 the script directory.

 ```bash
 cd ..
 ```

 Once you create and activate the environment, go ahead and install the
 packages

 ```bash
 pip install -r requirements.txt
 ```

Depending on the CUDA version (if any) and the operating system you are
running, you might need to manually install the suitable PyTorch version
[from here](https://pytorch.org/).


 ### Generate Graph Representation for Sentences

TBD

 ### Generate Knowledge Graph Embedding

We are using [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch)
to generate our knowledge graph embedding from ConceptNet. For details
on how to setup the environment or modify configurations, please visit
[OpenKE's home page](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch),
or see the same [README file](./kg-embeddings/OpenKE/README.md)
imported in this repository.

In short, before training the embeddings for the first time, you will
need to compile the model

```bash
cd kg-embeddings/OpenKE
bash make.sh
```

Then, you can move to the parent directory and begin training the
embeddings

```bash
cd ..
python run.py
```

The output of this script should be a 300d TransH embedding of
ConceptNet, which will be stored at `./data/embeddings/conceptnet/`
as a binary file.

To convert that binary file into the format that can be read by AllenNLP
later on, run the following script

```bash
python ./data_preprocessing/process-transh.py
```

Also, the previously downloaded ConceptNet-PPMI embedding also needs to
be process before it can be used by AllenNLP. To process
ConceptNet-PPMI, run the following script:

```bash
python ./data_preprocessing/process-conceptnet-ppmi.py
```

## Training the Models

Once everything are set up, you can start training the models by running
the following scripts

```bash
bash train.sh
```

This will train all the models, log the relevant information to stdout,
and store the serialized models to `./res`

To run each of the model individually, execute the following command:

```bash
allennlp train <model_config> --include-package code -s <stored_directory>
```

e.g.:
```bash
allennlp train config/graph.json --include-package code -s res/graph
```

You can view the configuration files in the [`./config/`](./config/)
directory. Here are the list of available models

| Model       | Paper     | Config  |
| ----------- | --------- | ------- |
| DecompAttn  | [A Decomposable Attention Model for Natural Language Inference (Parikh et al, 2016)](https://aclweb.org/anthology/D16-1244) | [`decompattn.json`](./config/decompattn.json) |
| MatchLSTM   | [Learning Natural Language Inference with LSTM (Wang and Jiang, 2016)](http://www.aclweb.org/anthology/N16-1170) | [`matchlstm.json`](./config/matchlstm.json) |
| DeIsTe      | [End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions (Yin et al, 2018)](http://aclweb.org/anthology/P18-2086) | [`deiste.json`](./config/deiste.json) |
| SimpleGraph | N/A       | [`graph.json`](./config/graph.json) |
| DecompAttn + Graph | N/A       | [`decompattn_graph.json`](./config/decompattn_graph.json) |
| MatchLSTM + Graph | N/A       | [`matchlstm_graph.json`](./config/matchlstm_graph.json) |
