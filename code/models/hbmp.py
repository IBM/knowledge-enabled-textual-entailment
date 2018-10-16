# modify from: https://github.com/Helsinki-NLP/HBMP
from typing import Dict, Optional

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import replace_masked_values, get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("hbmp")
class HBMP(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 rnn1: Seq2SeqEncoder,
                 rnn2: Seq2SeqEncoder,
                 rnn3: Seq2SeqEncoder,
                 aggregate_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(HBMP, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.rnn3 = rnn3
        self._aggregate_feedforward = aggregate_feedforward

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def encode(self, sentence, mask):
        out1, (ht1, ct1) = self.rnn1(sentence)
        # max pool
        emb1, _ = replace_masked_values(out1,
                                        mask.unsqueeze(-1),
                                        -1e7).max(dim=1)
        out2, (ht2, ct2) = self.rnn2(sentence, (ht1, ct1))
        # max pool
        emb2, _ = replace_masked_values(out2,
                                        mask.unsqueeze(-1),
                                        -1e7).max(dim=1)
        out3, (ht3, ct3) = self.rnn3(sentence, (ht2, ct2))
        emb3, _ = replace_masked_values(out3,
                                        mask.unsqueeze(-1),
                                        -1e7).max(dim=1)
        return torch.cat([emb1, emb2, emb3], dim=-1)

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        premise = self._text_field_embedder(premise)
        hypothesis = self._text_field_embedder(hypothesis)

        premise = self.encode(premise, premise_mask)
        hypothesis = self.encode(hypothesis, hypothesis_mask)

        aggregate_input = torch.cat(
            [premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis], 1)

        output_dict = {
            "final_hidden": aggregate_input,
        }

        if self._aggregate_feedforward:
            label_logits = self._aggregate_feedforward(aggregate_input)
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'HBMP':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)

        encoder_params = params.pop("encoder")
        rnn1 = Seq2SeqEncoder.from_params(encoder_params.duplicate())
        rnn2 = Seq2SeqEncoder.from_params(encoder_params.duplicate())
        rnn3 = Seq2SeqEncoder.from_params(encoder_params.duplicate())

        aggregated_params = params.pop('aggregate_feedforward', None)
        aggregate_feedforward = FeedForward.from_params(
            aggregated_params) if aggregated_params else None

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   rnn1=rnn1,
                   rnn2=rnn2,
                   rnn3=rnn3,
                   aggregate_feedforward=aggregate_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # add label to output
        argmax_indices = output_dict['label_probs'].max(dim=-1)[1].data.numpy()
        output_dict['label'] = [self.vocab.get_token_from_index(x, namespace="labels")
                                for x in argmax_indices]
        # do not show last hidden layer
        del output_dict["final_hidden"]
        return output_dict
