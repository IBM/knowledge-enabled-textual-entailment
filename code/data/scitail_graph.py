from typing import Dict, List
from overrides import overrides
import logging
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__)


@DatasetReader.register("scitail_graph")
class ScitailGraphDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 max_length: int=None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 entities_tokenizer: Tokenizer = None,
                 entities_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self._entities_tokenizer = entities_tokenizer
        self._entities_indexers = entities_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self._max_length = max_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)

            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                premise, premise_entities, hypothesis, hypothesis_entities, label = line.split(
                    '\t')
                yield self.text_to_instance(premise, premise_entities,
                                            hypothesis, hypothesis_entities,
                                            label)

    @overrides
    def text_to_instance(self, premise: str, premise_entities: str,
                         hypothesis: str, hypothesis_entities: str,
                         label: str = None) -> Instance:

        fields = {}

        if label:
            fields['label'] = LabelField(label)

        if self._tokenizer:
            premise = self._tokenizer.tokenize(premise)
            hypothesis = self._tokenizer.tokenize(hypothesis)

            if self._max_length:
                premise = premise[:self._max_length]
                hypothesis = hypothesis[:self._max_length]

            fields["premise"] = TextField(premise, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis, self._token_indexers)

        if self._entities_tokenizer:
            premise_entities = self._entities_tokenizer.tokenize(
                premise_entities)
            hypothesis_entities = self._entities_tokenizer.tokenize(
                hypothesis_entities)

            fields["premise_entities"] = TextField(premise_entities,
                                                   self._entities_indexers)
            fields["hypothesis_entities"] = TextField(hypothesis_entities,
                                                      self._entities_indexers)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ScitailGraphDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer_params = params.pop('tokenizer', None)
        entities_tokenizer_params = params.pop('entities_tokenizer', None)

        max_length = params.pop("max_length", None)

        if not tokenizer_params and not entities_tokenizer_params:
            raise ConfigurationError(
                "Please specify at least one of tokenizer and entities_tokenizer")
        tokenizer = Tokenizer.from_params(
            tokenizer_params) if tokenizer_params else None

        entities_tokenizer = Tokenizer.from_params(
            entities_tokenizer_params) if entities_tokenizer_params else None

        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        entities_indexers = TokenIndexer.dict_from_params(
            params.pop('entities_indexers', {}))

        params.assert_empty(cls.__name__)

        return cls(lazy=lazy, tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   max_length=max_length,
                   entities_tokenizer=entities_tokenizer,
                   entities_indexers=entities_indexers)


@WordSplitter.register('entities_splitter')
class EntitySplitter(WordSplitter):
        # split entities by comma
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t.strip().replace(" ", "_")) for t in sentence.split(',')]

    @classmethod
    def from_params(cls, params: Params) -> 'WordSplitter':
        params.assert_empty(cls.__name__)
        return cls()
