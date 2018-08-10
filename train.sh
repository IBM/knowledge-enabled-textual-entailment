#!/bin/bash


# Training DecompAttn
allennlp train config/decompattn.json -s res/decompattn --include-package code --file-friendly-logging

# Training Graph Model
allennlp train config/graph.json -s res/graph --include-package code --file-friendly-logging

# Training MatchLSTM
allennlp train config/matchlstm.json -s res/matchlstm --include-package code --file-friendly-logging

# Train DecompAttn with Graph
allennlp train config/decompattn_graph.json -s res/decompattn_graph --include-package code --file-friendly-logging

# Training MatchLSTM with Graph
allennlp train config/matchlstm_graph.json -s res/matchlstm_graph --include-package code --file-friendly-logging