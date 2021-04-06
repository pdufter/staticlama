from abc import ABC, abstractmethod
from typing import Text, List, Any, Set, Text, Union, Dict, TextIO, Callable
from utils.utils import load_triples, get_logger, load_templates, get_all_elements
from embedder import Embeddings
from transformers import AutoTokenizer, AutoModelWithLMHead, BertTokenizer, AutoConfig, BertConfig
import os
import json
import copy
import torch
import functools
import argparse
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


LOG = get_logger(__name__)


class Predictor(ABC):
    """Base class for doing typed prediction using either pretrained language models or static embeddings.
    """

    def __init__(self, tokenizer: Union[Text, AutoTokenizer], model_type: Text):
        """Summary

        Args:
            tokenizer (Union[Text, AutoTokenizer])
            model_type (Text): indicates whether we use a BERT model or a different model (using AutoModel)
        """
        self.model_type = model_type
        if isinstance(tokenizer, str):
            # CHANGE
            if self.model_type == "bert":
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        super(Predictor, self).__init__()

    @abstractmethod
    def typed_prediction(self, triples: List[Dict[Text, Text]], object_set: Set[Text], template: Text = None, n: int = 1) -> List[List[Text]]:
        """Summary

        Args:
            triples (List[Dict[Text, Text]]): triples for a relation containing subject and object
            object_set (Set[Text]): the set of all objects in the gold dataset
            template (Text, optional): a template, e.g., "[X] is the capital of [Y]"
            n (int, optional): get the top-n predictions.
        """
        pass


class LMPredictor(Predictor):

    """Get typed predictions from a pretrained language model.
    """

    def __init__(self, model: Union[Text, AutoModelWithLMHead], tokenizer: Union[Text, AutoTokenizer], model_type: Text,
                 device: Text = "cuda", use_templates: bool = True, output_hidden_states: bool = False):
        """Summary

        Args:
            model (Union[Text, AutoModelWithLMHead]): which PLM to use.
            tokenizer (Union[Text, AutoTokenizer]): corresponding tokenizer.
            model_type (Text): ndicates whether we use a BERT model or a different model (using AutoModel)
            device (Text, optional): e.g., "cuda"
            use_templates (bool, optional): Whether to use templates or replace them with the empty string.
            output_hidden_states (bool, optional): output hidden states of intermediate layers (they can then be directly used for prediction)
        """
        if isinstance(model, str):
            self.model_name = model.lower()
            config = AutoConfig.from_pretrained(model, output_hidden_states=output_hidden_states)
            self.model = AutoModelWithLMHead.from_pretrained(model, config=config)
        else:
            self.model_name = model.__class__.__name__.lower()
            self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.use_templates = use_templates
        super(LMPredictor, self).__init__(tokenizer, model_type)

    @staticmethod
    def is_valid_template(template: Text) -> bool:
        """
        Args:
            template (Text): the template

        Returns:
            bool:
        """
        return ("[X]" in template and "[Y]" in template)

    def fill_template(self, template: Text, subjects: List[Text], n_masks: int) -> List[Text]:
        """Fills templates, e.g.,  "[X] is the capital of [Y]." -> "Paris is the capital of [MASK]."

        Args:
            template (Text): Description
            subjects (List[Text]): Description
            n_masks (int): Description

        Returns:
            List[Text]: Description

        Raises:
            ValueError: Description
        """
        if not self.is_valid_template(template):
            LOG.warning("Invalid template: {}".format(template))
        else:
            filled_templates = []
            for subject in subjects:
                if not self.use_templates:
                    template = "[X] [Y]"
                filled_template = template.replace("[X]", subject)
                if self.model_type == "bert":
                    masks = " ".join(["[MASK]"] * n_masks)
                elif self.model_type == "xlm-r":
                    masks = " ".join(["<mask>"] * n_masks)
                else:
                    raise ValueError(f"Unknown model type {self.model_type}")
                filled_template = filled_template.replace("[Y]", masks)
                filled_templates.append(filled_template)
            return filled_templates

    def get_prediction_probabilities(self, texts: List[Text], n_mask_tokens: int) -> torch.Tensor:
        '''
        Output tensor shape: [batch_size, n_mask_tokens, vocabulary_size]

        Args:
            texts (List[Text]): Description
            n_mask_tokens (int): Description

        Returns:
            torch.Tensor: Description
        '''
        inputs = self.tokenizer.batch_encode_plus(texts, pad_to_max_length=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        mask_indices = [x == self.tokenizer.mask_token_id for x in inputs["input_ids"]]
        outputs = self.model(**inputs)
        predictions = outputs[0][torch.stack(mask_indices), :].reshape(
            (len(texts), n_mask_tokens, -1)).detach().cpu().numpy()
        return predictions

    def untokenize(self, tokens: List[Text]) -> Text:
        """
        Args:
            tokens (List[Text]): Description

        Returns:
            Text: Description

        Raises:
            NotImplementedError: Description
        """
        result = " ".join(tokens)
        if self.model_type == "bert":
            result = result.replace(" ##", "")
        else:
            raise NotImplementedError("'untokenize' only implemented for BERT models.")
        return result

    def get_type_accuracy(self, triples: List[Dict[Text, Text]], object_set: Set[Text], template: Text, n: int = 1, batch_size: int = 32) -> Set[Text]:
        """Investigate how often BERT predicts the correct type for a relation. 
        Get's all predicted objects in an untyped query setting.

        Args:
            triples (List[Dict[Text, Text]]): Description
            object_set (Set[Text]): Description
            template (Text): Description
            n (int, optional): Description
            batch_size (int, optional): Description

        Returns:
            Set[Text]: Description
        """
        object_set = list(object_set)
        object_set_tokenized = [self.tokenizer.tokenize(obj) for obj in object_set]
        ntokens_to_objects = collections.defaultdict(list)
        ntokens_to_objects_original = collections.defaultdict(list)
        for obj_original, obj in zip(object_set, object_set_tokenized):
            ntokens_to_objects[len(obj)].append(obj)
            ntokens_to_objects_original[len(obj)].append(obj_original)
        # TODO set max_k to 1 to restrict it to single tokens
        max_k = max([len(obj) for obj in object_set_tokenized])
        max_k = 1
        predictions = np.zeros((len(triples), len(object_set)))
        pred_tokens = collections.defaultdict(list)
        for i in range(0, len(triples), batch_size):
            triples_batch = triples[i:i + batch_size]
            obj_preds = []
            for k in range(1, max_k + 1):
                # create templates and feed to model
                filled_templates = self.fill_template(template, [triple["sub_label"] for triple in triples_batch], k)
                # get log probabilities for each k and for each subject
                probabilities = self.get_prediction_probabilities(filled_templates, k)
                # do "greedy decoding"
                for l in range(k):
                    for m in range(probabilities.shape[0]):
                        pred_tokens[(i + m, k)].append(self.tokenizer.convert_ids_to_tokens([probabilities[m, l, :].argmax()])[0])

        pred_tokens_set = set()
        for k, v in pred_tokens.items():
            pred_tokens_set.add(self.untokenize(v))
        return pred_tokens_set

    def typed_prediction(self, triples: List[Dict[Text, Text]], object_set: Set[Text], template: Text, n: int = 1, batch_size: int = 32) -> List[List[Text]]:
        """Do typed prediction, i.e., get prediction probabilities for all elements in 'object_set' and pick the one with 
        highest average log probability.

        Args:
            triples (List[Dict[Text, Text]]): Description
            object_set (Set[Text]): Description
            template (Text): Description
            n (int, optional): Description
            batch_size (int, optional): Description

        Returns:
            List[List[Text]]: top n predictions for each triple
        """
        object_set = list(object_set)
        object_set_tokenized = [self.tokenizer.tokenize(obj) for obj in object_set]
        ntokens_to_objects = collections.defaultdict(list)
        ntokens_to_objects_original = collections.defaultdict(list)
        for obj_original, obj in zip(object_set, object_set_tokenized):
            ntokens_to_objects[len(obj)].append(obj)
            ntokens_to_objects_original[len(obj)].append(obj_original)
        max_k = max([len(obj) for obj in object_set_tokenized])
        predictions = np.zeros((len(triples), len(object_set)))
        for i in range(0, len(triples), batch_size):
            triples_batch = triples[i:i + batch_size]
            obj_preds = []
            for k in range(1, max_k + 1):
                # create templates and feed to model
                current_subjects = [triple["sub_label"] for triple in triples_batch]
                filled_templates = self.fill_template(template, current_subjects, k)
                if any(["August Gailit" in x for x in filled_templates]):
                    import ipdb
                    ipdb.set_trace()
                # get log probabilities for each k and for each subject
                probabilities = self.get_prediction_probabilities(filled_templates, k)
                # for each subject get probabilities for each object
                for obj in ntokens_to_objects[k]:
                    obj_ids = self.tokenizer.convert_tokens_to_ids(obj)
                    obj_probs = probabilities[:, list(range(len(obj_ids))), obj_ids].mean(axis=1)
                    obj_preds.append(obj_probs)
                    # are those log probabilities?
            predictions[i:i + batch_size] = np.array(obj_preds).transpose()
        object_order = []
        for k in range(1, max_k + 1):
            object_order.extend(ntokens_to_objects_original[k])
        #object_order = np.array([x[0] for x in object_order])
        object_order = np.array(object_order)
        word_predictions = []
        for line in predictions.argsort(axis=1):
            word_predictions.append(object_order[line[::-1]].tolist()[:n])
        return word_predictions


class StaticPredictor(Predictor):

    """Get typed predictions from static embeddings.
    """

    def __init__(self, embeddings: Union[Text, Embeddings], tokenizer: Union[Text, AutoTokenizer], model_type: Text):
        """Summary

        Args:
            embeddings (Union[Text, Embeddings]): path to embeddings in standard word2vec/fasttext format.
            tokenizer (Union[Text, AutoTokenizer]): the tokenizer belonging to the embeddings.
            model_type (Text): Description
        """
        if isinstance(embeddings, str):
            self.embed = Embeddings()
            self.embed.load(embeddings)
            self.embed.get_mappings()
        else:
            self.embed = embeddings
        super(StaticPredictor, self).__init__(tokenizer, model_type)

    def extract_wordspace(self, target_vocab: List[Text]) -> Embeddings:
        """Extract embeddings for a target vocabulary. If a token is not in the embedding vocabulary
        mean pooling is used. 

        Args:
            target_vocab (List[Text]): Description

        Returns:
            Embeddings: the reduced embedding space that contains embeddings for each token in 'target_vocab'

        Raises:
            KeyError: Description
            ValueError: Description
        """
        if not isinstance(target_vocab, list):
            raise ValueError("target_vocab must be a list.")
        X = []
        for word in target_vocab:
            tokens = self.tokenizer.tokenize(word)
            token_indices = []
            try:
                for token in tokens:
                    if token in self.embed.word2index:
                        token_indices.append(self.embed.word2index[token])
                #token_indices = [self.embed.word2index[x] for x in tokens]
            except KeyError:
                raise KeyError("Token missing in static embedding space. Vocabulary mismatch?")
            if token_indices:
                Xtmp = self.embed.X[token_indices].mean(axis=0)
            else:
                Xtmp = np.zeros_like(self.embed.X[0])
            X.append(Xtmp)
        result = Embeddings()
        result.W = target_vocab
        result.Wset = set(target_vocab)
        result.X = np.array(X)
        return result

    def typed_prediction(self, triples: List[Dict[Text, Text]], object_set: Set[Text], template: Text = None, n: int = 1, measure: Text = "cosine", normalize: bool = False, add_relation_vector: bool = False, add_template_to_subject: bool = False) -> List[List[Text]]:
        """Do typed prediction, i.e., get the nearest neighbours for each object in the triples.

        For untyped prediction it should work to set object_set to the whole vocabulary of the language model.

        Args:
            triples (List[Dict[Text, Text]]): Description
            object_set (Set[Text]): Description
            template (Text, optional): Description
            n (int, optional): Description
            measure (Text, optional): which similarity measure to use. 
            normalize (bool, optional): whether to l2-normalize embeddings.
            add_relation_vector (bool, optional): if true, add the mean-pooled embedding vector of the template to the object.
            add_template_to_subject (bool, optional): mean-pool the subject and template to get a query vector, results in different weighting compared to 'add_relation_vector'.

        Returns:
            List[List[Text]]: top n predictions for each triple

        Raises:
            ValueError: Description
        """
        # get all subjects
        object_set = list(object_set)
        subject_set = []
        for triple in triples:
            if add_template_to_subject:
                subject_set.append(template.replace("[X]", triple["sub_label"]).replace("[Y]", "[MASK]").strip())
            else:
                subject_set.append(triple["sub_label"])

        queryspace = self.extract_wordspace(subject_set)
        keyspace = self.extract_wordspace(object_set)
        if add_relation_vector:
            # relation_space = self.extract_wordspace(template.replace("[X]", "").replace("[Y]", "").strip().split())
            relation_space = self.extract_wordspace([template])
            relation_vector = relation_space.X.mean(axis=0)
        else:
            relation_vector = np.zeros(queryspace.X.shape[1])
        if normalize:
            queryspace.normalize()
            keyspace.normalize()
        if measure == "cosine":
            sim = cosine_similarity(queryspace.X + relation_vector, keyspace.X)
        elif measure == "csls":
            sim = get_csls(queryspace.X + relation_vector, keyspace.X)
        else:
            raise ValueError("Unknown similarity measure: {}.".format(measure))
        argsim = sim.argsort(axis=1)[:, -n:]
        result = []
        for i in range(argsim.shape[0]):
            result.append([])
            for j in range(argsim.shape[1]):
                result[-1].append(keyspace.W[argsim[i, -(j + 1)]])
        return result


class MajorityPredictor(object):
    """Always predict the most frequent object in the object set. 
    """

    def __init__(self):
        """
        """
        pass

    def typed_prediction(self, triples: List[Dict[Text, Text]], object_set: Set[Text], template: Text = None, n: int = 1) -> List[List[Text]]:
        """

        Args:
            triples (List[Dict[Text, Text]]): Description
            object_set (Set[Text]): Description
            template (Text, optional): Description
            n (int, optional): Description

        Returns:
            List[List[Text]]: top n predictions for each triple
        """
        # get all subjects
        object_counter = collections.Counter()
        for triple in triples:
            object_counter[triple["obj_label"]] += 1
        result = []
        for i in range(len(triples)):
            result.append([])
            result[-1].extend([x[0] for x in object_counter.most_common(n)])
        return result


class LMStaticPredictor(StaticPredictor, LMPredictor, object):
    """Extract a static embedding space from a pretrained language model 
    and do simple similarity search. 
    """

    def __init__(self, layer: int, *args, **kwargs):
        """
        Args:
            layer (int): which layer to use in order to extract the static embedding space.
            *args: Description
            **kwargs: Description
        """
        #super(LMStaticPredictor, self).__init__(*args, **kwargs)
        self.layer = layer
        self.batch_size = 32
        LMPredictor.__init__(self, *args, **kwargs, output_hidden_states=True)

    def extract_wordspace(self, target_vocab: List[Text]) -> Embeddings:
        """Extract the static wordspace. Prediction functions can be used from parent class.

        Args:
            target_vocab (List[Text]): Description

        Returns:
            Embeddings: Description

        Raises:
            ValueError: Description
        """
        if not isinstance(target_vocab, list):
            raise ValueError("target_vocab must be a list.")
        X = []
        for i in range(0, len(target_vocab), self.batch_size):
            words_batch = target_vocab[i:i + self.batch_size]
            inputs = self.tokenizer.batch_encode_plus(words_batch, pad_to_max_length=True, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            outputs = self.model(**inputs)
            outputs[1][self.layer][~inputs["attention_mask"].type(torch.bool)] = 0
            Xtmp = outputs[1][self.layer].mean(axis=1).detach().cpu().numpy()
            X.extend(list(Xtmp))
        result = Embeddings()
        result.W = target_vocab
        result.Wset = set(target_vocab)
        result.X = np.array(X)
        return result


def get_csls(source: np.ndarray, target: np.ndarray, k: int = 5) -> np.ndarray:
    """Get CSLS as described in https://arxiv.org/pdf/1710.04087.pdf

    Args:
        source (np.ndarray): Description
        target (np.ndarray): Description
        k (int, optional): Description

    Returns:
        np.ndarray: Description
    """
    sim = cosine_similarity(source, target)
    argsim_source = sim.argsort(axis=1)[:, -k:]
    argsim_target = sim.argsort(axis=0)[-k:, :]
    avgsim_source = np.take_along_axis(sim, argsim_source, axis=1).mean(axis=1)
    avgsim_target = np.take_along_axis(sim, argsim_target, axis=0).mean(axis=0)
    return 2 * sim - avgsim_source.reshape((-1, 1)) - avgsim_target.reshape((1, -1))


class Evaluator(object):
    """Given a Predictor class evaluate the predictions on TREx and GoogleRE
    """

    def __init__(self, part: Text):
        """

        Args:
            part (Text): whether to use TREx or GoogleRE or both.
        """
        super(Evaluator, self).__init__()
        self._langs = ["ca", "az", "en", "ar", "uk", "fa", "tr", "it", "el", "ru", "hr", "hi", "sv", "sq", "fr", "ga", "eu", "de", "nl", "et", "he", "es", "bn", "ms", "sr",
                       "hy", "ur", "hu", "la", "sl", "cs", "af", "gl", "fi", "ro", "ko", "cy", "th", "be", "id", "pt", "vi", "ka", "ja", "da", "bg", "zh", "pl", "lv", "sk", "lt", "ta", "ceb"]
        self._relations = []
        if "trex" in part:
            self._relations = self._relations + ["P1412", "P108", "P178", "P31", "P36", "P407", "P449", "P127", "P364", "P106", "P176", "P937", "P463", "P138", "P101", "P39", "P530", "P264", "P1376",
                                                 "P1001", "P495", "P527", "P1303", "P190", "P47", "P30", "P361", "P103", "P20", "P27", "P279", "P19", "P159", "P413", "P37", "P140", "P740", "P276", "P136", "P17", "P131", ]
        # CHANGE
        if "googlere" in part:
            self._relations = self._relations + ["place_of_birth", "place_of_death"]

    def get_languages(self) -> List[Text]:
        return self._langs

    def get_relations(self) -> List[Text]:
        return self._relations

    @staticmethod
    def compute_precision(triples: List[Dict[Text, Text]], predictions: List[List[Text]], k: int) -> float:
        """
        Args:
            triples (List[Dict[Text, Text]]): Triples with subject and object.
            predictions (List[List[Text]]): The prediction for each triple in the same order as 'triples'.
            k (int): compute precision@k

        Returns:
            float: precision@k

        Raises:
            ValueError: Description
        """
        if len(triples) != len(predictions):
            raise ValueError("Groundtruth and prediction lengths do not match: {} vs. {}".format(
                len(triples), len(predictions)))
        nhits = 0
        for triple, prediction in zip(triples, predictions):
            if triple["obj_label"] in prediction:
                nhits += 1
        return nhits / len(triples)

    def evaluate(self, path: Text, langs: List[Text], relations: List[Text], predict: Callable, model_name: Text, outfile: Text = None, n: int = 5, mode: Text = "predict", relevant_ids: Dict = None) -> Dict[Text, Dict[Text, float]]:
        """
        Args:
            path (Text): path to the dataset. 
            langs (List[Text]): which languages to evaluate on.
            relations (List[Text]): which relations to evaluate on.
            predict (Callable): prediction function to get predictions (e.g., Predictor.typed_prediction)
            model_name (Text): deprecated.
            outfile (Text, optional): where to store results.
            n (int, optional): evaluate precision@n
            mode (Text, optional): do prediction or measure how well a BERT model is for typing
            relevant_ids (Dict, optional): consider only triples with relevant_ids (e.g., for LAMA-UHN) 

        Returns:
            Dict[Text, Dict[Text, float]]: All sorts of results.
        """
        if outfile is not None:
            if not os.path.exists(outfile):
                outfile = open(outfile, "w")
            else:
                LOG.warning("Outfile {} exists. Results are not stored.".format(outfile))
        details = collections.defaultdict(lambda: collections.defaultdict(dict))
        results = collections.defaultdict(lambda: collections.defaultdict(dict))
        for lang in langs:
            # CHANGE
            # templates = load_templates(os.path.join(path, "templates/relations_{}.jsonl".format(lang)))
            templates = load_templates(os.path.join(path, "{}/templates.jsonl".format(lang)))
            for relation in relations:
                LOG.info("Processing {}-{}".format(lang, relation))
                if relevant_ids:
                    # triples = load_triples(path, lang, relation, filter_english=True, filter_ids=relevant_ids[relation])
                    triples = load_triples(path, lang, relation, filter_english=False,
                                           filter_ids=relevant_ids[relation])
                else:
                    # CHANGE
                    # triples = load_triples(path, lang, relation, filter_english=True)
                    triples = load_triples(path, lang, relation, filter_english=False)
                objects = get_all_elements(triples, lambda x: [x])
                objects = [x[0] for x in objects]
                predictions = predict(triples, objects, templates[relation]["template"], n)
                if mode == "predict":
                    prec = self.compute_precision(triples, predictions, n)
                    details[lang][relation]["triples"] = triples
                    details[lang][relation]["objects"] = objects
                    details[lang][relation]["predictions"] = predictions
                    details[lang][relation]["p@{}".format(n)] = prec
                    results[lang][relation] = prec
                    LOG.info("Precision {}".format(prec))
                elif mode == "evaltyping":
                    recall = len(set(objects) & predictions) / len(set(objects))
                    precision = len(set(objects) & predictions) / len(set(predictions))
                    if (precision + recall) != 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    details[lang][relation]["predictions"] = list(predictions)
                    details[lang][relation]["objects"] = list(objects)
                    details[lang][relation]["predictions&objects"] = list(predictions & set(objects))
                    details[lang][relation]["predictions-objects"] = list(predictions - set(objects))
                    details[lang][relation]["objects-predictions"] = list(set(objects) - predictions)
                    details[lang][relation]["recall{}".format(n)] = recall
                    details[lang][relation]["precision{}".format(n)] = precision
                    details[lang][relation]["F1{}".format(n)] = f1
                    results[lang][relation] = recall
                    LOG.info("Recall {}".format(recall))
                    LOG.info("Precision {}".format(precision))
                    LOG.info("F1 {}".format(f1))
        for lang in details:
            LOG.info("p@{} {} {}".format(n, lang, np.mean([details[lang]
                                                           [relation]["p@{}".format(n)] for relation in details[lang]])))
        if outfile:
            json.dump(details, outfile)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, required=True, help="path to mlama data")
    parser.add_argument("--part", default="trex", type=str, required=False, help="trex or googlere")
    parser.add_argument("--relations", default="all", type=str, required=False, help="comma separated list of relation names or 'all'")
    parser.add_argument("--similaritymeasure", default="cosine", type=str, required=False, help="csls or cosine")
    parser.add_argument("--details", default=None, type=str, required=False, help="path to store the detailed results")
    parser.add_argument("--summary", default=None, type=str, required=True, help="path to write the overview results")
    parser.add_argument("--prefix", default="", type=str, required=False, help="prefix for results, such as experiment id")
    parser.add_argument("--lm", default=None, type=str, required=False, help="name or path for the language model")
    parser.add_argument("--lmstatic", default=None, type=int, required=False, help="if passing an integer, extract static embedding space from this layer")
    parser.add_argument("--model_type", default="bert", type=str, required=False, help="which model to use: bert or automodel")
    parser.add_argument("--lang", default="en", type=str, required=False, help="language to evaluate on")
    parser.add_argument("--embeddings", default=None, type=str, required=False, help="path to static embeddings")
    parser.add_argument("--idfilter", default=None, type=str, required=False, help="path to a file with relevant lineids")
    parser.add_argument("--vocab", default=None, type=str, required=False, help="path to the tokenizer config files")
    parser.add_argument("--topn", default=1, type=int, required=False, help="consider top n predictions")
    parser.add_argument("--normalize", action="store_true", help="if true, l2-normalize embeddings")
    parser.add_argument("--majority", action="store_true", help="run majority baseline")
    parser.add_argument("--do_not_use_templates", action="store_true", help="ignore templates")
    parser.add_argument("--add_relationvector", action="store_true", help="add relation vector to static embeddings")
    parser.add_argument("--add_template_to_subject", action="store_true", help="use template when querying static embeddings")
    parser.add_argument("--mode", default="predict", type=str, required=False, help="predict or evaltyping")
    args = parser.parse_args()

    to_predict = []
    if args.lm:
        if args.lmstatic is not None:
            lm = LMStaticPredictor(args.lmstatic, args.lm, args.lm, args.model_type)
            if args.mode == "predict":
                emb_typed_prediction = functools.partial(
                    lm.typed_prediction, measure=args.similaritymeasure, normalize=args.normalize, add_relation_vector=args.add_relationvector, add_template_to_subject=args.add_template_to_subject)
                to_predict.append((args.lm, emb_typed_prediction))
            else:
                raise NotImplementedError()
        else:
            lm = LMPredictor(args.lm, args.lm, args.model_type, use_templates=not args.do_not_use_templates)
            if args.mode == "predict":
                to_predict.append((args.lm, lm.typed_prediction))
            else:
                to_predict.append((args.lm, lm.get_type_accuracy))

    if args.embeddings and args.vocab:
        emb = StaticPredictor(args.embeddings, args.vocab, args.model_type)
        emb_typed_prediction = functools.partial(
            emb.typed_prediction, measure=args.similaritymeasure, normalize=args.normalize, add_relation_vector=args.add_relationvector, add_template_to_subject=args.add_template_to_subject)
        to_predict.append((args.embeddings, emb_typed_prediction))

    if args.majority:
        maj = MajorityPredictor()
        to_predict.append(("majority", maj.typed_prediction))

    if args.idfilter != "all":
        with open(args.idfilter) as fp:
            relevant_ids = json.load(fp)
    else:
        relevant_ids = collections.defaultdict(list)
    outfile = open(args.summary, "a")
    for model_name, prediction_function in to_predict:
        ev = Evaluator(args.part)
        if args.relations == "all":
            relations = ev._relations
        else:
            relations = args.relations.split(",")
        result = ev.evaluate(args.data, [args.lang], relations, prediction_function,
                             model_name, n=args.topn, outfile=args.details, mode=args.mode, relevant_ids=relevant_ids)
        for lang in result:
            for relation in result[lang]:
                outfile.write("{} {} {} {} {}\n".format(
                    args.prefix, model_name, lang, relation, result[lang][relation]))
    outfile.close()


if __name__ == '__main__':
    main()
