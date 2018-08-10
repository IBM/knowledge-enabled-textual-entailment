# An example to generate the prediction script
allennlp predict res/matchlstm/model.tar.gz --include-package code \
    --predictor nli_predictor \
     "./data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed_twohop/entities/scitail_1.0_dev.jsonl"\
     > matchlstm_predict.jsonl