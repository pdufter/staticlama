from nltk import sent_tokenize
from tqdm import tqdm
import argparse


def main():
    """Prepare wikipedia coming out of wikiextractor and optionally sentence tokenize it using NLTK.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, type=str, required=True, help="The corpus file.")
    parser.add_argument("--outfile", default=None, type=str, required=True, help="The output file.")
    parser.add_argument("--sentence_tokenize", action="store_true", help="")
    args = parser.parse_args()

    with open(args.infile, "r") as fin:
        with open(args.outfile, "w") as fout:
            for line in tqdm(fin):
                line = line.strip()
                if line.startswith("<doc ") and line.endswith(">"):
                    continue
                if line == "</doc>":
                    continue
                if line:
                    if args.sentence_tokenize:
                        sentences = sent_tokenize(line)
                    else:
                        sentences = [line]
                    for sentence in sentences:
                        fout.write("{}\n".format(sentence))


if __name__ == '__main__':
    main()
