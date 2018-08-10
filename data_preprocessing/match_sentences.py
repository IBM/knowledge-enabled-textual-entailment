# Match the processed sentence back to original premise-hypothsis pairs
import os
from pathlib import Path

# directory that contains the current file (to make sure this script can be run
# from anywhere without messinbg up the relative path)
CURR_DIR = Path(os.path.dirname(__file__))
# location of the original dataset
DATASET_DIR = CURR_DIR / '../data/nli_datasets/SciTailV1/tsv_format'
# location of the pased dataset (with conceptnet)
ENTITIES_DIR = DATASET_DIR / 'conceptnet_processed_twohop/entities'

# location of the matched and processed dataset (output directory)
PARSED_DIR = ENTITIES_DIR

# file to store sentences that cannot be parsed
ERROR_DIR = PARSED_DIR / 'no_entities'

splits = ['train', 'dev', 'test']

if not PARSED_DIR.exists():
    PARSED_DIR.mkdir()

if not ERROR_DIR.exists():
    ERROR_DIR.mkdir()

# load original string-entities pair
sen_entities = {}
entities_file = ENTITIES_DIR / "scitail_sentences_conceptnet_entities.tsv"

with entities_file.open(encoding="utf-8", errors="ignore") as f:
    for line in f:
        sentence, entities = line.strip("\n").split("\t")
        sen_entities[sentence] = entities

for split in splits:

    # match and create the new file
    dataset_file = DATASET_DIR / "scitail_1.0_{}.tsv".format(split)
    output_file = PARSED_DIR / "scitail_1.0_{}.tsv".format(split)
    error_file = ERROR_DIR / "scitail_1.0_{}.tsv".format(split)
    with dataset_file.open(encoding="utf-8", errors="ignore") as f:
        with output_file.open("w", encoding="utf-8", errors="ignore") as parsed_file:
            with error_file.open("w", encoding="utf-8", errors="ignore") as err:
                for line in f:
                    premise, hypothesis, label = line.strip("\n").split("\t")

                    if premise in sen_entities and hypothesis in sen_entities:
                        print("{}\t{}\t{}\t{}\t{}".format(premise,
                                                          sen_entities[premise],
                                                          hypothesis,
                                                          sen_entities[hypothesis],
                                                          label),
                              file=parsed_file
                              )
                    else:
                        print(line, file=err)
