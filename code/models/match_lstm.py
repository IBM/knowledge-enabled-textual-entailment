# the MatchLSTM model, rewritten in AllenNLP
from typing import Optional, Dict
import torch
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("match_lstm")
class MatchLSTM(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 inter_attention: MatrixAttention,
                 output_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MatchLSTM, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._inter_attention = inter_attention
        self._output_feedforward = output_feedforward

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        # Embed premise and hypothesis
        premise = self._text_field_embedder(premise)  # (n x p x d)
        hypothesis = self._text_field_embedder(hypothesis)  # (n x h x d)

        # encode premise and hypothesis
        # (n x p x 2d) if bidirectional else (n x p x d)
        premise = self._encoder(premise, premise_mask)
        # (n x h x 2d) if bidirectional else (n x h x d)
        hypothesis = self._encoder(hypothesis, hypothesis_mask)

        # calculate matrix attention
        similarity_matrix = self._inter_attention(hypothesis,
                                                  premise)  # (n x h x p)

        attention_softmax = last_dim_softmax(similarity_matrix,
                                             premise_mask)  # (n x h x p)
        hypothesis_tilda = weighted_sum(
            premise, attention_softmax)  # (n x h x 2d) assuming encoder is bidirectional

        hypothesis_matching_states = torch.cat([hypothesis,
                                                hypothesis_tilda,
                                                hypothesis - hypothesis_tilda,
                                                hypothesis * hypothesis_tilda], dim=-1)

        # max pool
        hypothesis_max, _ = replace_masked_values(hypothesis_matching_states,
                                                  hypothesis_mask.unsqueeze(-1),
                                                  -1e7).max(dim=1)  # (n x 2d)

        output_dict = {"final_hidden": hypothesis_max}

        if self._output_feedforward:
            label_logits = self._output_feedforward(hypothesis_max)
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            output_dict["label_logits"] = label_logits
            output_dict["label_probs"] = label_probs

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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MatchLSTM':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        inter_attention = MatrixAttention.from_params(
            params.pop("inter_attention"))

        output_feedforward_params = params.pop('output_feedforward')
        output_feedforward = FeedForward.from_params(
            output_feedforward_params) if output_feedforward_params else None

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   inter_attention=inter_attention,
                   output_feedforward=output_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
