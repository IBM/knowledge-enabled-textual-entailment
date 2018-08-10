#!/bin/bash

# Evaluating DecompAttn
allennlp evaluate res/decompattn/model.tar.gz --include-package code --evaluation-data-file \
    ../CSNLI/data/nli_datasets/scitail/conceptnet_entities/parsed/scitail_1.0_test_conceptnet_entities.tsv

# Evaluating graph model
allennlp evaluate res/graph/model.tar.gz --include-package code --evaluation-data-file \
    ../CSNLI/data/nli_datasets/scitail/conceptnet_entities/parsed/scitail_1.0_test_conceptnet_entities.tsv

# Evaluating MatchLSTM
allennlp evaluate res/matchlstm/model.tar.gz --include-package code --evaluation-data-file \
    ../CSNLI/data/nli_datasets/scitail/conceptnet_entities/parsed/scitail_1.0_test_conceptnet_entities.tsv

# Evaluating DecompAttn with Graph
allennlp evaluate res/decompattn_graph/model.tar.gz --include-package code --evaluation-data-file \
    ../CSNLI/data/nli_datasets/scitail/conceptnet_entities/parsed/scitail_1.0_test_conceptnet_entities.tsv

# Evaluating MatchLSTM with Graph
allennlp evaluate res/matchlstm_graph/model.tar.gz --include-package code --evaluation-data-file \
    ../CSNLI/data/nli_datasets/scitail/conceptnet_entities/parsed/scitail_1.0_test_conceptnet_entities.tsv
