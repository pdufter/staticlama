# StaticLAMA: Static Embeddings as Efficient Knowledge Bases?

This repository contains code for the NAACL 2021 paper ["Static Embeddings as Efficient Knowledge Bases?"]().
We investigate how static word embeddings can be used on a modified LAMA probe.

## Embeddings

The pretrained static embeddings are available [here](). The pretrained contextualized embeddings can be downloaded using
[huggingface](https://huggingface.co/).

## Reproduction

To reproduce our results:

### 1. Create conda environment and install requirements

(optional) It might be a good idea to use a separate conda environment. It can be created by running:
```
conda create -n staticlama -y python=3.7 && conda activate staticlama
pip install -r requirements.txt
```

add project to path:

export PYTHONPATH=${PYTHONPATH}:/path-to-project

### 2. Download the data


```bash
wget http://cistern.cis.lmu.de/mlama/mlama.zip
unzip mlama.zip
rm data.zip
```

### 3. Run the experiments

```bash
python scripts/run_experiments_mBERT_ranked.py --lang "fr"
python scripts/eval.py
```

## Reference:

```bibtex
@inproceedings{dufter2021static,
    title = "Static Embeddings as Efficient Knowledge Bases?",
    author = {Dufter, Philipp  and
      Kassner, Nora  and
      Sch{\"u}tze, Hinrich},
    booktitle = "to appear in Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}


@inproceedings{kassner2021multilingual,
    title = "Multilingual {LAMA}: Investigating Knowledge in Multilingual Pretrained Language Models",
    author = {Kassner, Nora  and
      Dufter, Philipp  and
      Sch{\"u}tze, Hinrich},
    booktitle = "to appear in Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Acknowledgements

tbd


## License

MIT



