"""
Train a knowledge graph embedding for concepnet
"""
from OpenKE.config import Config
from OpenKE import models
from utils import download, APP_ROOT, TQDM_DISABLE
import logging
from pathlib import Path
import gzip
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s -"
                    " %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_concepnet(force=False):
    """
    @brief      Downloads a concepnet, if not already.

    @param      force  if True, ignore existing file and re-download the content

    @return     path of the downloaded gzip csv file of concepnet
    """
    CONCEPNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz"
    CONCEPNET_PATH = APP_ROOT / "../data/conceptnet/conceptnet-assertions-5.6.0.csv.gz"

    # Download the concepnet dataset, if not already
    if not force and CONCEPNET_PATH.exists():
        return CONCEPNET_PATH

    logger.info("Downloading ConcepNet...")
    concepnet_path = download(CONCEPNET_URL, CONCEPNET_PATH)
    return concepnet_path


def csv_to_tuples(path, out_path=None, force=False):
    """
    @brief      Transform the ConceptNet csv files to the toples format
    that OpenKE takes as input

    @param      path  the path of the file
    @param      out_path the path to store the tuples. If None, then the
    tuples will be stored in the same directory as the csv file
    @param      force  if True, ignore existing file and re-process the csv

    @return     the path to the parent directory of the tuple files
    """

    ENTITY_ID_FILE = "entity2id.txt"
    RELATION_ID_FILE = "relation2id.txt"
    TUPLE_FILE = "train2id.txt"
    CONCEPNET_SIZE = 32755210

    def is_english(entity):
        """
        @brief      Check whether an entity is in English
        @param      entity  The id of the entity
        @return     True if the entity is in english, False otherwise.
        """
        return entity[:6] == b"/c/en/"

    # Structures used to map entity and relation names to their ids
    entities = {}
    relations = {}
    tuples = []

    if out_path is None:
        out_path = Path(path).parent
    else:
        out_path = Path(out_path)

    if not force and (out_path / TUPLE_FILE).exists() and (out_path / RELATION_ID_FILE).exists() \
            and (out_path / ENTITY_ID_FILE).exists():
        # abort if file has already been processed
        return out_path

    logger.info("Converting CSV to tuples to {}...".format(out_path.resolve()))

    with gzip.open(path, "rb") as gzip_file:
        for line in tqdm(gzip_file, disable=TQDM_DISABLE, total=CONCEPNET_SIZE):
            # The columns are tab-separated
            relation, entity1, entity2 = line.split(b'\t')[1:4]
            # Only preserve the English relationships
            if is_english(entity1) and is_english(entity2):
                # extract the name from id
                entity1, entity2 = (token_id.split(b'/')[3].decode()
                                    for token_id in (entity1, entity2))
                relation = relation.split(b'/')[2].decode()
                # Generate new id for entities and relations that appear for the first time
                if relation not in relations:
                    relations[relation] = len(relations)
                for entity in (entity1, entity2):
                    if entity not in entities:
                        entities[entity] = len(entities)

                tuples.append((entities.get(entity1), entities.get(entity2),
                               relations.get(relation)))

    logger.info(
        "Conversion completed, found {} English pairs.".format(len(tuples)))

    with (out_path / TUPLE_FILE).open("w") as tuple_file:
        print(len(tuples), file=tuple_file, end="")
        # write the relation to the file. Print newline at the beginning of the line
        # so that the file won't be ended with a new line character
        for entity1, entity2, relation in tuples:
            print("\n{} {} {}".format(entity1, entity2,
                                      relation), file=tuple_file, end="")

    logger.info(
        "Dumping {} relation ids and {} entity ids".format(len(relations), len(entities)))

    # dump the ids
    with (out_path / ENTITY_ID_FILE).open("w") as entity_id_file:
        print(len(entities), file=entity_id_file, end="")
        for entity, entity_id in entities.items():
            # Similar as above. Print newline at the beginning of the line
            # so that the file won't be ended with a new line character
            print("\n{}\t{}".format(entity, entity_id), file=entity_id_file, end="")

    with (out_path / RELATION_ID_FILE).open("w") as relation_id_file:
        print(len(relations), file=relation_id_file, end="")
        for relation, relation_id in relations.items():
            print("\n{}\t{}".format(relation, relation_id),
                  file=relation_id_file, end="")

    return out_path


def run():
    # Download and preprocess ConcepNet
    csv_path = download_concepnet()
    tuples_directory = csv_to_tuples(csv_path)

    config = Config()
    config.set_in_path(f'{tuples_directory}/')
    #config.set_in_path(r'/Users/ashishnagar/code/knowledge-enabled-textual-entailment/kg-embeddings/OpenKE/benchmarks/FB15K/')
    print(tuples_directory)
    config.set_log_on(1)  # set to 1 to print the loss

    config.set_work_threads(8)
    config.set_train_times(1)  # number of iterations
    config.set_nbatches(2)  # batch size
    config.set_alpha(0.001)  # learning rate

    config.set_bern(0)
    config.set_dimension(100)
    config.set_margin(1.0)
    config.set_ent_neg_rate(1)
    config.set_rel_neg_rate(0)
    config.set_opt_method("SGD")

    
    OUTPUT_PATH = APP_ROOT / "../data/embeddings/conceptnet/"
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()

    OUTPUT_PATH = str(OUTPUT_PATH)
    print('Output Path : {}'.format(OUTPUT_PATH))
    # Model parameters will be exported via torch.save() automatically.
    config.set_export_files(OUTPUT_PATH + "/transh.pt")
    # Model parameters will be exported to json files automatically.
    # (Might cause IOError if the file is too large)
    config.set_out_files(OUTPUT_PATH + "/transh_embedding.vec.json")

    config.init()
    # Save the graph embedding every {number} iterations
    config.set_export_steps(1)

    config.set_model(models.TransH)

    logger.info("Begin training with {}".format(config.__dict__))

    config.set_import_files(r'/Users/ashishnagar/code/knowledge-enabled-textual-entailment/data/embeddings/conceptnet/transh.pt.meta')
    saved_model = config.restore_tensorflow()
    print(config.get_parameters_by_name('ent_embeddings'))

if __name__ == "__main__":
    run()
