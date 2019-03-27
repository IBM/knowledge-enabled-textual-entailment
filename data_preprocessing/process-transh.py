# Convert TransH from binary weights to Word2Vec format

from pathlib import Path
import torch
import pandas as pd
# manually install tahbles via `pip install tables` if python complains
# about missing tables module:
# `HDFStore requires PyTables, "No module named 'tables'" problem importing`

# get file path
DATASET_DIR = Path(__file__).parent / '../data/embeddings/conceptnet'
FILE_PATH = DATASET_DIR / 'transh.pt'

# load pre-trained transH model
transh = torch.load(FILE_PATH, map_location='cpu')

with open(Path(__file__).parent / "../data/conceptnet/entity2id.txt") as f:
    next(f)  # skip header
    entity2id = [word_pair.strip().split()[0] for word_pair in f.readlines()]

embed = transh.get("ent_embeddings.weight").numpy()
# use pandas to insert index
out = pd.DataFrame(embed, index=entity2id)

# save the processed file
out.to_csv(path_or_buf=DATASET_DIR / 'transh.txt.gz', sep=' ',
           header=False, compression="gzip")