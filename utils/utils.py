from transformers import BertTokenizer, BertForMaskedLM
import torch
from typing import Tuple, Callable, Text, List, Set, Dict
import collections
import argparse
import logging
import json
from tqdm import tqdm
import os


def get_logger(name, filename=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


LOG = get_logger(__name__)


def load_triples(path: Text, lang: Text, relation: Text, filter_english: bool = True, filter_ids: Set[int] = None) -> List[Dict]:
    """
    Args:
        path (Text): path to dataset.
        lang (Text): which language to load.
        relation (Text): which relation to load.
        filter_english (bool, optional): if True, use only triples that are translated and not copied from English.
        filter_ids (Set[int], optional): filter the dataset for relevant ids ('lineid').

    Returns:
        List[Dict]: List of triples.
    """
    filepath = os.path.join(path, lang, relation + ".jsonl")
    if not os.path.exists(filepath):
        LOG.warning("{} does not exist. Choose different language or relation.".format(filepath))
        return []
    else:
        triples = []
        with open(filepath, "r") as fp:
            for i, line in enumerate(fp):
                if line.strip():
                    triple = json.loads(line.strip())
                    if (not filter_english or not triple["from_english"]) and (not filter_ids or i in filter_ids):
                        triples.append(triple)
    return triples


def load_templates(path: Text) -> Dict[Text, Dict]:
    """
    Args:
        path (Text): path to the templates in json format.

    Returns:
        Dict[Text, Dict]: templates.
    """
    templates = {}
    with open(path) as fp:
        for line in fp:
            template = json.loads(line)
            templates[template["relation"]] = template
    return templates


def get_all_elements(triples: List[Dict], tokenize: Callable, field_type: Text = "obj_label", filter_english: bool = False) -> Set[Text]:
    """Extracts all elements with a given field_type, e.g., all objects.

    Args:
        triples (List[Dict]): List of triples.
        tokenize (Callable): tokenizer for the extracted field_type.
        field_type (Text, optional): which field_type to use.
        filter_english (bool, optional): if True, use only triples that are translated and not copied from English.

    Returns:
        Set[Text]: Set of all elements with the given field type.
    """
    valid = set()
    for triple in triples:
        if not filter_english or not triple["from_english"]:
            true = triple[field_type]
            true_tokenized = tuple(tokenize(true))
            valid.add(true_tokenized)
    return valid
