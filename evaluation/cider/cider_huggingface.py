# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""
import math
from collections import defaultdict
from typing import Dict, Any

import datasets
import evaluate
import numpy as np

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"

def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n)

class CiderScorer(object):
    """CIDEr scorer.
    """

    def __init__(self, refs, test=None, n=4, sigma=6.0, doc_frequency=None, ref_len=None):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.doc_frequency = defaultdict(float)
        self.ref_len = None

        for k in refs.keys():
            self.crefs.append(cook_refs(refs[k]))
            if test is not None:
                self.ctest.append(cook_test(test[k][0]))  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

        if doc_frequency is None and ref_len is None:
            # compute idf
            self.compute_doc_freq()
            # compute log reference length
            self.ref_len = np.log(float(len(self.crefs)))
        else:
            self.doc_frequency = doc_frequency
            self.ref_len = ref_len

    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.doc_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.doc_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self):
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)


# class Cider:
#     """
#     Main Class to compute the CIDEr metric
#
#     """
#     def __init__(self, gts=None, n=4, sigma=6.0):
#         # set cider to sum over 1 to 4-grams
#         self._n = n
#         # set the standard deviation parameter for gaussian penalty
#         self._sigma = sigma
#         self.doc_frequency = None
#         self.ref_len = None
#         if gts is not None:
#             tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma)
#             self.doc_frequency = tmp_cider.doc_frequency
#             self.ref_len = tmp_cider.ref_len
#
#     def compute_score(self, gts, res):
#         """
#         Main function to compute CIDEr score
#         :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
#                 res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
#         :return: cider (float) : computed CIDEr score for the corpus
#         """
#         assert(gts.keys() == res.keys())
#         cider_scorer = CiderScorer(gts, test=res, n=self._n, sigma=self._sigma, doc_frequency=self.doc_frequency,
#                                    ref_len=self.ref_len)
#         return cider_scorer.compute_score()
#
#     def __str__(self):
#         return 'CIDEr'


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Cider(evaluate.Metric):
    """TODO: Short description of my metric."""

    def __init__(self, gts=None, n=4, sigma=6.0, **kwargs):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.doc_frequency = None
        self.ref_len = None
        if gts is not None:
            tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma)
            self.doc_frequency = tmp_cider.doc_frequency
            self.ref_len = tmp_cider.ref_len
        super().__init__(**kwargs)

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("string")),
                # "references": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "references": datasets.Sequence(datasets.Value("string")),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        pass

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        assert(gts.keys() == res.keys())
        cider_scorer = CiderScorer(gts, test=res, n=self._n, sigma=self._sigma, doc_frequency=self.doc_frequency,
                                   ref_len=self.ref_len)
        return cider_scorer.compute_score()

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        """Returns the scores"""

        # gts = {k: [' '.join(w) for w in v] for k, v in enumerate(references)}
        # res = {k: [' '.join(v)] for k, v in enumerate(predictions)}
        gts = {k: v for k, v in enumerate(references)}
        res = {k: v for k, v in enumerate(predictions)}
        score, scores = self.compute_score(gts, res)

        if kwargs.get('return_scores'):
            return {self.name: scores}
        return {self.name: score}