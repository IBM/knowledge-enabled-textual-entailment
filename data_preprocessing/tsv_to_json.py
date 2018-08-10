# Match the processed sentence back to original premise-hypothsis pairs
import os
from pathlib import Path
import json

# directory that contains the current file (to make sure this script can be run
# from anywhere without messinbg up the relative path)
CURR_DIR = Path(os.path.dirname(__file__))
# location of the original dataset
ENTITIES_DIR = CURR_DIR / \
    '../data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed_twohop/entities'

# location of the matched and processed dataset (output directory)
PARSED_DIR = ENTITIES_DIR

splits = ['dev']

if not PARSED_DIR.exists():
    PARSED_DIR.mkdir()

for split in splits:
    # match and create the new file
    dataset_file = ENTITIES_DIR / "scitail_1.0_{}.tsv".format(split)
    output_file = PARSED_DIR / "scitail_1.0_{}.jsonl".format(split)
    with dataset_file.open(encoding="utf-8", errors="ignore") as f:
        with output_file.open("w", encoding="utf-8", errors="ignore") as parsed_file:
            for line in f:
                premise, premise_entities, hypothesis,\
                    hypothesis_entities, label = line.strip(
                        "\n").split("\t")
                instance = {
                    "premise": premise,
                    "premise_entities": premise_entities,
                    "hypothesis": hypothesis,
                    "hypothesis_entities": hypothesis_entities,
                    "label": label
                }
                json.dump(instance, parsed_file)
                parsed_file.write("\n")
