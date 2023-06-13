"""
Adapted from:

https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py

"""

from collections import namedtuple as _namedtuple
from typing import List

import spacy as _spacy
from os import system as _system


class Fragments(object):
    Match = _namedtuple("Match", ("summary", "text", "length"))

    @classmethod
    def _load_model(cls):

        if not hasattr(cls, "_en"):

            try:

                cls._en = _spacy.load("en_core_web_sm")  # change tokenizer

            except:

                _system("python3 -m spacy download en_core_web_sm")
                cls._en = _spacy.load("en_core_web_sm")

        if not hasattr(cls, "_de"):

            try:

                cls._de = _spacy.load("de_core_news_sm")  # change tokenizer

            except:

                _system("python3 -m spacy download de_core_news_sm")
                cls._de = _spacy.load("de_core_news_sm")

    def __init__(self, summary, text, language="english", tokenize=True, case=False):

        self._load_model()

        self._tokens = tokenize

        self.summary = self._tokenize(summary, language) if tokenize else summary.split()
        self.text = self._tokenize(text, language) if tokenize else text.split()

        self._norm_summary = self._normalize(self.summary, case)
        self._norm_text = self._normalize(self.text, case)

        self._match(self._norm_summary, self._norm_text)

    def _tokenize(self, text, language):

        """

        Tokenizes input using the fastest possible SpaCy configuration.
        This is optional, can be disabled in constructor.

        """

        if language == "english":
            return self._en(text, disable=["tagger", "parser", "ner", "textcat"])
        elif language == "german":
            return self._de(text, disable=["tagger", "parser", "ner", "textcat"])
        else:
            return NotImplementedError

    def _normalize(self, tokens, case=False):

        """

        Lowercases and turns tokens into distinct words.

        """

        return [
            str(t).lower()
            if not case
            else str(t)
            for t in tokens
        ]

    def overlaps(self):

        """

        Return a list of Fragments.Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):

            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment

        """

        return self._matches

    def get_metric(self, metric_name, summary_base=True, text_to_summary=True):
        if metric_name == "coverage":
            return self.coverage(summary_base=summary_base)
        elif metric_name == "density":
            return self.density(summary_base=summary_base)
        elif metric_name == "compression":
            return self.compression(text_to_summary=text_to_summary)
        else:
            raise NotImplementedError

    def coverage(self, summary_base=True):

        """

        Return the COVERAGE score of the summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal COVERAGE score within [0, 1]

        """

        numerator = sum(o.length for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.reference)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def density(self, summary_base=True):

        """

        Return the DENSITY score of summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal DENSITY score within [0, ...]

        """

        numerator = sum(o.length ** 2 for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.reference)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def compression(self, text_to_summary=True):

        """

        Return compression ratio between summary and text.

        Arguments:

            - text_to_summary (bool): compute text/summary ratio (default = True)

        Returns:

            - decimal compression score within [0, ...]

        """

        ratio = [len(self.text), len(self.summary)]

        try:

            if text_to_summary:
                return ratio[0] / ratio[1]
            else:
                return ratio[1] / ratio[0]

        except ZeroDivisionError:

            return 0

    def _match(self, a, b):

        """

        Raw procedure for matching summary in text, described in paper.

        """

        self._matches = []

        a_start = b_start = 0

        while a_start < len(a):

            best_match = None
            best_match_length = 0

            while b_start < len(b):

                if a[a_start] == b[b_start]:

                    a_end = a_start
                    b_end = b_start

                    while a_end < len(a) and b_end < len(b) \
                            and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1

                    length = a_end - a_start

                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length

                    b_start = b_end

                else:

                    b_start += 1

            b_start = 0

            if best_match:

                if best_match_length > 0:
                    self._matches.append(best_match)

                a_start += best_match_length

            else:

                a_start += 1


def extraction_analysis(
    source: List,
    summary: List,
    language: str = "english",
    prefix: str = "",
    metrics_to_compute: List = ["coverage"],
    summary_base: bool = True,
    text_to_summary: bool = True
):
    metrics = {metric_name: [] for metric_name in metrics_to_compute}

    for source_text, summary_text in zip(source, summary):

        fragments = Fragments(summary_text, source_text, language=language)
        for metric_name in metrics_to_compute:
            metric_value = fragments.get_metric(
                metric_name=metric_name,
                summary_base=summary_base,
                text_to_summary=text_to_summary
            )
            metrics[metric_name].append(metric_value)

    metrics = {f"{prefix}_{metric_name}": metric_values for metric_name, metric_values in metrics.items()}

    return metrics
