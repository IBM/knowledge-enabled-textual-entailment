# ConceptNet-PPMI Embeddings can be downloaded here: https://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/16.09/conceptnet-55-ppmi.h5

import os
from pathlib import Path
import pandas as pd
# manually install tahbles via `pip install tables` if python complains
# about missing tables module:
# `HDFStore requires PyTables, "No module named 'tables'" problem importing`

# get file path
DATASET_DIR = Path(os.path.dirname(__file__)) / '../data/embeddings/conceptnet'
FILE_PATH = DATASET_DIR / 'conceptnet-55-ppmi.h5'

data = pd.read_hdf(FILE_PATH)

# remove non-english words
eng = data.filter(regex="^/c/en/", axis=0)

# remove prefix
eng.index = eng.index.map(lambda idx: idx[6:])


OUTPUT_PATH = DATASET_DIR / 'conceptnet-55-ppmi-en.txt.gz'

# save to the same format as word2vec and glove
eng.to_csv(path_or_buf=OUTPUT_PATH, sep=' ', header=False, compression="gzip")
