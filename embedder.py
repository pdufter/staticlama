import os
from typing import Text, Any, List, Iterable
from tqdm import tqdm
import collections
from utils.utils import get_logger
import numpy as np
LOG = get_logger(__name__)


class Embeddings(object):
    """Class to load, edit and store word embeddings.

    Attr:
        X: embedding matrix
        W: list of words
        Wset: set of words
    """

    def __init__(self) -> None:
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        pass

    def load(self, path: Text, load_first_n: int = None, header: bool = True) -> None:
        """Load word embeddings in word2vec format from a txt file.

        Args:
            path: path to the embedding file
            load_first_n: int; how many lines to load
            header: bool; whether the embedding file contains a header line
        """
        self.path = path
        LOG.info("loading embeddings: {}".format(self.path))

        fin = open(self.path, 'r')

        if header:
            n, d = map(int, fin.readline().split())
        else:
            n, d = None, None

        data = {}
        count = 0
        for line in tqdm(fin):
            count += 1
            if load_first_n is not None and count > load_first_n:
                break
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

        self.W = list(data.keys())
        self.Wset = set(self.W)
        self.X = np.vstack(tuple([data[x] for x in self.W]))

        LOG.info("loaded {} / {} vectors with dimension {}.".format(len(self.W), n, self.X.shape[1]))

    def normalize(self) -> None:
        """Normalize the embeddings with l2 norm
        """
        self.X = (self.X.transpose() / np.linalg.norm(self.X, axis=1)).transpose()

    def filter(self, relevant: Iterable[bool]) -> None:
        """Filter the embeddings to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        choose = []
        for word in self.W:
            if word in relevant:
                choose.append(True)
            else:
                choose.append(False)
        self.W = list(np.array(self.W)[choose])
        self.Wset = set(self.W)
        self.X = self.X[choose]

        LOG.info("filtered for {} / {} words.".format(len(relevant), len(self.W)))

    def store(self, fname: Text) -> None:
        """Store the embedding space

        Args:
            fname: path to the file
        """
        outfile = open(fname, "w")
        n, dim = self.X.shape
        outfile.write("{} {}\n".format(n, dim))
        for i in range(n):
            outfile.write(self.W[i])
            for k in range(dim):
                outfile.write(" {}".format(self.X[i, k]))
            outfile.write("\n")
        outfile.close()

    def set_prefix(self, prefix: Text) -> None:
        self.W = [prefix + x for x in self.W]
        self.Wset = set([prefix + x for x in self.Wset])

    @staticmethod
    def replace_prefix(prefix: Text, word: Text) -> Text:
        return word.replace(prefix, "", 1)

    def remove_prefix(self, prefix: Text) -> None:
        self.W = [self.replace_prefix(prefix, x) for x in self.W]
        self.Wset = set([self.replace_prefix(prefix, x) for x in self.Wset])

    def get_mappings(self):
        self.index2word = {i: w for (i, w) in enumerate(self.W)}
        self.word2index = {w: i for (i, w) in enumerate(self.W)}


class EmbeddingTrainer(object):
    """docstring for Embedding"""

    def __init__(self, input_file: Text, output_dir: Text) -> None:
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_corpus(self):
        LOG.info("Writing corpus...")
        with open(os.path.join(self.output_dir, "corpus.txt"), "w") as fout, open(self.input_file, "r") as fin:
            for line in tqdm(fin):
                if line.strip():
                    tokenized = self.tokenizer.tokenize(line.strip())
                    fout.write("{}\n".format(" ".join(tokenized)))

    def get_vocabulary(self, vocabulary_size: int, tokenizer: Text) -> None:
        self.tokenizer = Tokenizer(tokenizer)
        self.tokenizer.train(self.input_file, vocabulary_size, self.output_dir)

    def train(self, dim: int, subwords: bool = False) -> None:
        if subwords:
            minn, maxn = 3, 6
        else:
            minn, maxn = 0, 0
        command = """
        nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
        -input {} \
        -output {} \
        -dim {} \
        -minCount 2 \
        -lr 0.025 \
        -epoch 15 \
        -neg 15 \
        -thread 48 \
        -minn {} \
        -maxn {}
        """.format(os.path.join(self.output_dir, "corpus.txt"),
                   os.path.join(self.output_dir, "embeddings"),
                   dim,
                   minn,
                   maxn)
        os.system(command)
