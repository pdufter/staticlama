# StaticLAMA: Static Embeddings as Efficient Knowledge Bases?

This repository contains code for the NAACL 2021 paper ["Static Embeddings as Efficient Knowledge Bases?"](https://arxiv.org/pdf/2104.07094.pdf).
We investigate how static word embeddings can be used on a modified LAMA probe.

This code contains functionality to answer knowledge intensive questions using basic similarity search with static embeddings
or using typed querying in pretrained language models. 

With some modifications it can also be used for the standard LAMA probe. 


## Embeddings

The pretrained static embeddings are available [here](http://cistern.cis.lmu.de/staticlama). The pretrained contextualized embeddings can be downloaded using
[huggingface](https://huggingface.co/).

## Running the code

To reproduce our results:

### 1. Create conda environment and install requirements

(optional) It might be a good idea to use a separate conda environment. It can be created by running:
```
conda create -n staticlama -y python=3.7 && conda activate staticlama
pip install -r requirements.txt
```

### 2. Setup dependencies

In order to run the experiments end to end, two external tools are required: 
1) [Wikiextractor](https://github.com/attardi/wikiextractor)
2) [fastText](https://github.com/facebookresearch/fastText)

Please install them and modify the paths to the binaries in `staticlama.sh`


### 3. Run the experiments

```bash
bash staticlama.sh
```

If you encounter any issue or want to improve the code, please reach out to us.

## References

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


## License

Copyright (C) 2020 Philipp Dufter

A full copy of the license can be found in LICENSE.



