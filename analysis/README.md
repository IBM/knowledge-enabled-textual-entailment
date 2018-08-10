#  Experiments

| Model              | Train   | Dev       | Note          | Config                                                         |
| ------------------ | ------- | --------- | ------------- | -------------------------------------------------------------- |
| Graph              | 0.79831 |  0.65391  | One Hop       | [`graph-triples.json`](./experiments/graph-triples.json)       |
| Graph              | 0.99572 |  0.73773  | Entities Only | [`graph-entities.json`](./experiments/graph-entities.json)     |
| DecompAttn + Graph | 0.99165 |  0.75153  |               | [`decompattn-graph.json`](./experiments/decompattn-graph.json) |
| DecompAttn         | 0.98474 |  0.75307  |               | [`decompattn.json`](./experiments/decompattn.json)             |
| MatchLSTM          | 1.0     |  0.84356  | 100d GloVe    | [`matchlstm-100d.json`](./experiments/matchlstm-100d.json)     |
| MatchLSTM + Graph  | 0.99606 |  0.86656  |               | [`matchlstm-graph.json`](./experiments/matchlstm-graph.json)   |
| MatchLSTM          | 0.99280 |**0.88594**|               | [`matchlstm.json`](./experiments/matchlstm.json)               |
