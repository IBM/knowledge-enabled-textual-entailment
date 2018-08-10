from typing import Dict, Optional
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("graph_and_text")
class GraphAndTextModel(Model):

    def __init__(self, vocab: Vocabulary,
                 classify_feed_forward: FeedForward,
                 text_model: Model=None,
                 graph_model: Model=None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(GraphAndTextModel, self).__init__(vocab, regularizer)

        self._graph_model = graph_model
        self._text_model = text_model
        self._classify_feed_forward = classify_feed_forward

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                premise_entities: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                hypothesis_entities: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:

        # Feed the data to text and graph model
        text_out = self._text_model(premise, hypothesis).get("final_hidden")

        graph_out = self._graph_model(premise_entities,
                                      hypothesis_entities).get("final_hidden")

        # combine the results (n x 4d)
        combined_input = torch.cat((text_out, graph_out), dim=-1)

        label_logits = self._classify_feed_forward(combined_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {
            "label_logits": label_logits,
            "label_probs": label_probs,
        }

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'GraphAndTextModel':
        text_model = params.pop("text_model", None)
        text_model = Model.from_params(vocab, text_model)

        graph_model = params.pop("graph_model")
        graph_model = Model.from_params(vocab, graph_model)

        classify_feed_forward = FeedForward.from_params(
            params.pop('classify_feed_forward'))

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   classify_feed_forward=classify_feed_forward,
                   text_model=text_model,
                   graph_model=graph_model,
                   initializer=initializer,
                   regularizer=regularizer)
