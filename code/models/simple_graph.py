from typing import Dict, Optional
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_graph")
class SimpleGraphModel(Model):

    def __init__(self, vocab: Vocabulary,
                 entities_embedder: TextFieldEmbedder,
                 inter_attention: MatrixAttention,
                 project_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleGraphModel, self).__init__(vocab, regularizer)

        self._entities_embedder = entities_embedder
        self._inter_attention = inter_attention
        self._project_feedforward = TimeDistributed(project_feedforward)
        self._aggregate_feedforward = aggregate_feedforward

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,
                premise_entities: Dict[str, torch.LongTensor],
                hypothesis_entities: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:

        # Premise embeddings
        premise = self._entities_embedder(premise_entities)  # (n x p x d)
        premise_mask = get_text_field_mask(premise_entities).float()

        # Hypothesis embeddings
        hypothesis = self._entities_embedder(
            hypothesis_entities)  # (n x h x d)
        hypothesis_mask = get_text_field_mask(hypothesis_entities).float()

        # Attention Mechanism
        # Weight Matrix
        e = self._inter_attention(premise, hypothesis)  # (n x p x h)

        # Compute alpha and beta
        alpha = last_dim_softmax(e, hypothesis_mask)  # (n x p x h)
        beta = last_dim_softmax(e.transpose(
            1, 2).contiguous(), premise_mask)  # (n x h x p)

        # Compute premise-tilda and hypothesis-tilda
        premise_attn = weighted_sum(hypothesis, alpha)  # (n x p x d)
        hypothesis_attn = weighted_sum(premise, beta)  # (n x h x d)

        # Merge Hypothesis + Hypothesis-Tilda and Premise + Premise-Tilda -- Generate Context-Hypothesis
        # and Context-Premise
        # premise = torch.cat((premise_encode, premise_attn,
        premise = torch.cat((premise, premise_attn,
                             premise - premise_attn,
                             premise * premise_attn),
                            dim=-1)  # (n x p x 4d)
        hypothesis = torch.cat((hypothesis, hypothesis_attn,
                                hypothesis - hypothesis_attn,
                                hypothesis * hypothesis_attn),
                               dim=-1)  # (n x h x 4d)

        # Projection for hypothesis and Premise using a feed forward layer
        premise = self._project_feedforward(premise)  # (n x p x d)
        hypothesis = self._project_feedforward(hypothesis)  # (n x h x d)

        # using mean- and max-pool to get a fix-size vector
        premise_max, _ = replace_masked_values(
            premise, premise_mask.unsqueeze(-1), -1e7).max(dim=1)  # (n x d)
        hypothesis_max, _ = replace_masked_values(
            hypothesis, hypothesis_mask.unsqueeze(-1), -1e7).max(dim=1)  # (n x d)
        premise_mean = (premise * premise_mask.unsqueeze(-1)).sum(dim=1) /  \
            premise_mask.sum(dim=1, keepdim=True)  # (n x d)
        hypothesis_mean = (hypothesis * hypothesis_mask.unsqueeze(-1)).sum(dim=1) /  \
            hypothesis_mask.sum(dim=1, keepdim=True)  # (n x d)

        premise = torch.cat((premise_mean, premise_max), dim=-1)  # (n x 2d)
        hypothesis = torch.cat(
            (hypothesis_mean, hypothesis_max), dim=-1)  # (n x 2d)

        # final prediction
        aggregate_input = torch.cat([premise, hypothesis], dim=-1)  # (n x 4d)

        output_dict = {"final_hidden": aggregate_input}

        if self._aggregate_feedforward:
            label_logits = self._aggregate_feedforward(
                aggregate_input)  # (n x 2)
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SimpleGraphModel':
        embedder_params = params.pop("entities_embedder")
        entities_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        inter_attention = MatrixAttention.from_params(
            params.pop("inter_attention"))
        project_feedforward = FeedForward.from_params(
            params.pop('project_feedforward'))

        aggregated_params = params.pop('aggregate_feedforward', None)
        aggregate_feedforward = FeedForward.from_params(
            aggregated_params) if aggregated_params else None

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   entities_embedder=entities_embedder,
                   inter_attention=inter_attention,
                   project_feedforward=project_feedforward,
                   aggregate_feedforward=aggregate_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
