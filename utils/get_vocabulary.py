from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
import argparse
from typing import Text
import os


def get_vocabulary(infile: Text, vocabsize: int, outfolder: Text):
    """Train a new BERT Wordpiece Vocabulary
    
    Args:
        infile (Text): path to file on which the vocabulary should be trained.
        vocabsize (int): the vocabulary size.
        outfolder (Text): where to store the trained vocabulary.
    """
    # get special token maps and config
    autotok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    autotok.save_pretrained(args.outfolder)
    os.remove(os.path.join(args.outfolder, "vocab.txt"))

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False, clean_text=False)

    # Then train it!
    tokenizer.train([args.infile], vocab_size=args.vocabsize, limit_alphabet=int(1e9))

    # And finally save it somewhere
    tokenizer.save(args.outfolder, "vocab")
    os.rename(os.path.join(args.outfolder, "vocab-vocab.txt"), os.path.join(args.outfolder, "vocab.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, type=str, required=True, help="")
    parser.add_argument("--outfolder", default=None, type=str, required=True, help="")
    parser.add_argument("--vocabsize", default=None, type=int, required=True, help="")
    args = parser.parse_args()
    get_vocabulary(args.infile, args.vocabsize, args.outfolder)
