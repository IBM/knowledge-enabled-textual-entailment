python -m allennlp.service.server_simple \
    --archive-path res/matchlstm/model.tar.gz \
    --predictor nli_predictor \
    --include-package code \
    --title "NLI Predictor" \
    --field-name premise \
    --field-name hypothesis