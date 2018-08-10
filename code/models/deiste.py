# the DeIsTe model, rewritten in AllenNLP
from typing import Optional, Dict
import torch
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("deiste")
class DeIsTe(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 inter_attention: MatrixAttention,
                 param_dyn_encoder: Seq2VecEncoder,
                 pos_embedder: TokenEmbedder,
                 pos_attn_encoder: Seq2VecEncoder,
                 output_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DeIsTe, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._inter_attention = inter_attention
        self._param_dyn_encoder = param_dyn_encoder
        self._pos_embedder = pos_embedder
        self._pos_attn_encoder = pos_attn_encoder
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

        matrix_mask = torch.bmm(premise_mask.unsqueeze(2),
                                hypothesis_mask.unsqueeze(1))  # (n x p x h)

        # Calculate interaction between sentences
        interaction = self._inter_attention(premise,
                                            hypothesis)  # (n x p x h)

        p_dyn = self.paramConv(premise, interaction, premise_mask,
                               matrix_mask)  # (n x d)
        h_dyn = self.paramConv(hypothesis, interaction.transpose(1, 2),
                               hypothesis_mask,
                               matrix_mask.transpose(1, 2))  # (n x d)

        p_pos = self.posAttnConv(premise, hypothesis, interaction,
                                 premise_mask, hypothesis_mask, matrix_mask)
        h_pos = self.posAttnConv(hypothesis, premise,
                                 interaction.transpose(1, 2).contiguous(),
                                 hypothesis_mask, premise_mask,
                                 matrix_mask.transpose(1, 2))

        combined_input = torch.cat((p_dyn, h_dyn, p_pos, h_pos),
                                   dim=-1)  # (n x 4d)

        output_dict = {"final_hidden": combined_input}

        if self._output_feedforward:
            label_logits = self._output_feedforward(combined_input)
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            output_dict["label_logits"] = label_logits
            output_dict["label_probs"] = label_probs

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def paramConv(self, sentence, interaction, sentence_mask, matrix_mask):
        """
        @brief      Calculate the parameter-dynamic convolution of the given
                    sentence

        @param      self         The object
        @param      sentence     The embeded sentence (n x s x d)
        @param      interaction  The interaction matrix (n x s x s')
        @param      sentence_mask  The mask of the sentence (n x s)
        @param      matrix_mask  The mask of the interaction matrix (n x s x s')

        @return     the convoluted representation of the sentence (n x d)
        """

        # calculate the important score of each word in the sentence
        max_i, _ = replace_masked_values(interaction, matrix_mask,
                                         -1e7).max(dim=-1)  # (n x s)
        alpha = 1 / (1 + max_i)  # (n x s)

        weighted_sentence = (sentence.permute(
            2, 0, 1) * alpha).permute(1, 2, 0)   # (n x s x d)

        return self._param_dyn_encoder(weighted_sentence, sentence_mask)

    def posAttnConv(self, sentence, other_sen, interaction, sentence_mask,
                    other_sen_mask, matrix_mask):
        """
        @brief      Compute the position-aware attentive convolution

        @param      self            The object
        @param      sentence        The embeded sentence (n x s x d)
        @param      other_sen       The other sentence (n x s' x d)
        @param      interaction     The interaction matrix (n x s x s')
        @param      sentence_mask   The mask of the sentence (n x s)
        @param      other_sen_mask  The mask of other sentence (n x s')
        @param      matrix_mask     The mask of the interaction matrix (n x s x
                                    s')

        @return     The position-aware attentive convolution
        """
        # calculate the representation of the sentence
        interaction_softmax = last_dim_softmax(
            interaction, other_sen_mask)  # (n x s x s')
        sentence_tilda = weighted_sum(
            other_sen, interaction_softmax)  # (n x s x d)

        # get index of the best-matched word
        _, x = replace_masked_values(interaction, matrix_mask,
                                     -1e7).max(dim=-1)  # (n x s)
        z = self._pos_embedder(x)  # (n x s x dm)

        sentence_combined = torch.cat((sentence_tilda, sentence, z),
                                      dim=2)  # (n x s x (2d + dm))

        return self._pos_attn_encoder(sentence_combined, sentence_mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DeIsTe':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)

        inter_attention = MatrixAttention.from_params(
            params.pop("inter_attention"))
        param_dyn_encoder = Seq2VecEncoder.from_params(
            params.pop("param_dyn_encoder"))

        pos_embedder = TokenEmbedder.from_params(
            vocab=None, params=params.pop("pos_embedder"))
        pos_attn_encoder = Seq2VecEncoder.from_params(
            params.pop("pos_attn_encoder"))

        output_feedforward_params = params.pop('output_feedforward', None)
        output_feedforward = FeedForward.from_params(
            output_feedforward_params) if output_feedforward_params else None

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   inter_attention=inter_attention,
                   param_dyn_encoder=param_dyn_encoder,
                   pos_embedder=pos_embedder,
                   pos_attn_encoder=pos_attn_encoder,
                   output_feedforward=output_feedforward,
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
