from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('nli_predictor')
class NLIPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        premise = json_dict.get('premise', None)
        hypothesis = json_dict.get('hypothesis', None)
        premise_entities = json_dict.get('premise_entities', None)
        hypothesis_entities = json_dict.get('hypothesis_entities', None)
        instance = self._dataset_reader.text_to_instance(
            premise=premise, hypothesis=hypothesis,
            premise_entities=premise_entities,
            hypothesis_entities=hypothesis_entities)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
