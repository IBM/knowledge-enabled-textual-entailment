import requests
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import math

APP_ROOT = Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
TQDM_DISABLE = logger.getEffectiveLevel() < logging.INFO


def download(url, path):
    """
    @brief      A helper function to download the file from {url} and store that
                to {path}

    @param      url   The url of the file to be downloaded
    @param      path  The path to store the downloaded file (type str or Path)

    @return     the absolute path of the downloaded file
    """
    path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    logger.info(f"Downloading from {url} to {path.resolve()}")

    res = requests.get(url, stream=True)

    # Information for displaying an progress-bar
    block_size = 1024
    total_size = int(res.headers.get('content-length', 0))

    with path.open("wb") as f:
        for chunk in tqdm(res.iter_content(chunk_size=block_size),
                          total=math.ceil(total_size / block_size),
                          unit="KB", disable=TQDM_DISABLE):
            if chunk:
                f.write(chunk)

    return path


def parse_args():
    """
    @brief      Parse the arguments that are entered by user via command line

    @return     a namespace of the parsed arguments from command line
    """
    parser = argparse.ArgumentParser()
    # parameters accepted by the parser
    parser.add_argument('-p', '--preprocess', dest='preprocess',
                        action='store_true', default=False, help='If specified,'
                        ' preprocess ConceptNet before training the model')
    parser.add_argument('-r', '--resume', action='store_true', dest='resume',
                        default=False, help='If specified, load the model'
                        ' from disk instead of initialize a new one')
    parser.add_argument('-b', '--batch_size', metavar='n', dest='batch_size',
                        type=int, default=50, help='Number of samples in each '
                        'batch')
    parser.add_argument('--hidden_size', metavar='d', dest='hidden_size',
                        type=int, default=300, help='Dimension of the hidden '
                        'layer')
    parser.add_argument('-l', '--learning_rate', metavar='l', dest='learning_rate',
                        type=float, default=0.01, help='Learning rate')
    parser.add_argument('-k', '--kernel_size', metavar='k', dest='kernel_size',
                        type=int, default=3, help='Kernel size of conv layer')
    parser.add_argument('--pos_emb_size', metavar='dm', dest='pos_emb_size',
                        type=int, default=50, help='Position embedding size')
    parser.add_argument('--min_freq', metavar='f', dest='min_freq',
                        type=int, default=10, help='Words that appear lower '
                        'than min_freq will be replaced by <unk>')
    parser.add_argument('--regulation', metavar='l2', dest='regulation',
                        type=float, default=1e-6, help='amount of l2 regulation')
    parser.add_argument('--use_mlog', dest='use_mlog', action='store_true',
                        default=False, help='whether to use meter logger or not')
    return parser.parse_args()
