"""
Adapted from:

https://github.com/neulab/guided_summarization/blob/master/scripts/sents.py

A script to get the oracle-selected sentences (separated by <q>)
The first argument is the path to source documents (sentences separated by <q>)
The second argument is the path to reference summaries (sentences seprated by <q>)
The third argument is the path to the output

Example command:

python3 dataset/guidance/sents.py \
    --src_path data/ge_wiki/test \
    --tgt_path data/ge_wiki/test \
    --src_file_extension .src \
    --tgt_file_extension .tgt \
    --output_path tmp
    
"""

import re
import sys
import nltk
import numpy as np

from pprint import pprint
from typing import List
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    from guidance_utils import extract_guidance
else:
    from .guidance_utils import extract_guidance


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    # abstract = _rouge_clean(' '.join(abstract)).split()
    # sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def guidance_sent_extract(
        src: str,
        tgt: str,
        tokenize: bool = True,
        language: str = "english"
) -> str:
    """ Extract guidance sentences for single source/summary pair. """

    src = src.rstrip()
    tgt = tgt.rstrip()
    src = sent_tokenize(src, language=language)
    tgt = sent_tokenize(tgt, language=language)

    if tokenize:
        src = [' '.join(nltk.tokenize.word_tokenize(x)) for x in src]
        tgt = [' '.join(nltk.tokenize.word_tokenize(x)) for x in tgt]

    oracle_ids = greedy_selection(src, tgt, 3)
    guidance_sentences = [src[i] for i in oracle_ids]
    guidance_text = '<q>'.join(guidance_sentences)

    return oracle_ids, guidance_text


def guidance_sents_extract(
        srcs: List[str],
        tgts: List[str],
        tokenize: bool = False,
        get_text: bool = True,
        language: str = "english"  # TODO: change language for german
) -> List[str]:
    """ Extract guidance sentences for multiple source/summary pair. """
    guidance_sentences = []
    for src, tgt in zip(srcs, tgts):
        guidance_sentences.append(
            guidance_sent_extract(src, tgt, tokenize, language)[int(get_text)]
        )
    return guidance_sentences


def get_extracted_summary(
        tgts: List,
        preds: List,
        block_trigram: bool = True
):
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    pred = []

    for filtered_text, sent_scores in zip(tgts, preds):

        sent_scores = np.array(sent_scores[:len(filtered_text)])
        selected_ids = np.argsort(-sent_scores)
        _pred = []

        for i in selected_ids:
            candidate = filtered_text[i].strip()
            if block_trigram:
                if not _block_tri(candidate, _pred):
                    _pred.append(candidate)
            else:
                _pred.append(candidate)

            if len(_pred) == 3:
                break

        pred.append(_pred)

    return pred


if __name__ == "__main__":
    # Imports
    sys.path.append(".")  # If run in summarization dir
    sys.path.append("../..")  # If run in summarization/dataset/guidance dir

    from utils.parser import ArgumentParser
    from utils.arguments import AbsSumSentenceGuidanceArguments

    # Parser
    hf_parser = ArgumentParser((AbsSumSentenceGuidanceArguments,))
    args, remaining = \
        hf_parser.parse_args_with_json_into_dataclasses_with_default(
            return_remaining_strings=True)
    hf_parser.check_no_remaining_args(remaining)

    print("### Arguments:\n")
    pprint(vars(args))
    extract_guidance(func=guidance_sents_extract, **vars(args))
