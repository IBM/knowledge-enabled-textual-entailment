# return hidden state and output
from torch import nn

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Seq2SeqEncoder


@Seq2SeqEncoder.register('rnn_encoder')
class RNNEncoder(Seq2SeqEncoder):

    def __init__(self, module: nn.Module) -> None:
        super(RNNEncoder, self).__init__()
        self._module = module

    def forward(self, sentence, hidden_state=None):
        return self._module(sentence, hidden_state)

    @classmethod
    def from_params(cls, params: Params) -> 'RNNEncoder':
        module = params.pop('module').lower()
        if module == 'lstm':
            module_class = nn.LSTM
        elif module == 'gru':
            module_class = nn.GRU
        elif module == 'rnn':
            module_class = nn.RNN
        else:
            raise ConfigurationError("Unsupported module type")

        module = module_class(**params.as_dict())
        return RNNEncoder(module)
