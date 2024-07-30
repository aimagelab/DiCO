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
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from typing import Dict, Any

import datasets
import evaluate
import numpy as np
# TODO: Add BibTeX citation
import requests

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

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_GZ_URL = '/path/to/spice.tgz'
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'


# class Spice:
#     """
#     Main Class to compute the SPICE metric
#     """
#
#     def __init__(self):
#         base_path = os.path.dirname(os.path.abspath(__file__))
#         jar_path = os.path.join(base_path, SPICE_JAR)
#         gz_path = os.path.join(base_path, os.path.basename(SPICE_GZ_URL))
#         if not os.path.isfile(jar_path):
#             if not os.path.isfile(gz_path):
#                 download_from_url(SPICE_GZ_URL, gz_path)
#             tar = tarfile.open(gz_path, "r")
#             tar.extractall(path=os.path.dirname(os.path.abspath(__file__)))
#             tar.close()
#             os.remove(gz_path)
#
#
#     def float_convert(self, obj):
#         try:
#             return float(obj)
#         except:
#             return np.nan
#
#     def compute_score(self, gts, res):
#         assert (sorted(gts.keys()) == sorted(res.keys()))
#         imgIds = sorted(gts.keys())
#
#         # Prepare temp input file for the SPICE scorer
#         input_data = []
#         for id in imgIds:
#             hypo = res[id]
#             ref = gts[id]
#
#             # Sanity check.
#             assert (type(hypo) is list)
#             assert (len(hypo) == 1)
#             assert (type(ref) is list)
#             assert (len(ref) >= 1)
#
#             input_data.append({
#                 "image_id": id,
#                 "test": hypo[0],
#                 "refs": ref
#             })
#
#         cwd = os.path.dirname(os.path.abspath(__file__))
#         temp_dir = os.path.join(cwd, TEMP_DIR)
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#         in_file = tempfile.NamedTemporaryFile('w+', delete=False, dir=temp_dir, encoding='utf8')
#         json.dump(input_data, in_file, indent=2)
#         in_file.close()
#
#         # Start job
#         out_file = tempfile.NamedTemporaryFile('w+', delete=False, dir=temp_dir, encoding='utf8')
#         out_file.close()
#         cache_dir = os.path.join(cwd, CACHE_DIR)
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#         spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
#                      '-cache', cache_dir,
#                      '-out', out_file.name,
#                      '-subset',
#                      '-silent'
#                      ]
#
#         try:
#             from subprocess import DEVNULL  # Python 3.
#         except ImportError:
#             DEVNULL = open(os.devnull, 'wb')
#         subprocess.check_call(spice_cmd,
#                               cwd=os.path.dirname(os.path.abspath(__file__)),
#                               stdout=DEVNULL, stderr=DEVNULL)
#
#         # Read and process results
#         with open(out_file.name) as data_file:
#             results = json.load(data_file)
#         os.remove(in_file.name)
#         os.remove(out_file.name)
#
#         imgId_to_scores = {}
#         spice_scores = []
#         for item in results:
#             imgId_to_scores[item['image_id']] = item['scores']
#             spice_scores.append(self.float_convert(item['scores']['All']['f']))
#         average_score = np.mean(np.array(spice_scores))
#         scores = []
#         for image_id in imgIds:
#             # Convert none to NaN before saving scores over subcategories
#             score_set = {}
#             for category, score_tuple in imgId_to_scores[image_id].items():
#                 score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
#             scores.append(score_set)
#         return average_score, scores
#
#     def __str__(self):
#         return 'SPICE'


def float_convert(obj):
    try:
        return float(obj)
    except:
        return np.nan


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Spice(evaluate.Metric):
    """TODO: Short description of my metric."""

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
        base_path = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(base_path, SPICE_JAR)
        gz_path = os.path.join(base_path, os.path.basename(SPICE_GZ_URL))
        if not os.path.isfile(jar_path):
            if not os.path.isfile(gz_path):
                download_from_url(SPICE_GZ_URL, gz_path)
            tar = tarfile.open(gz_path, "r")
            tar.extractall(path=os.path.dirname(os.path.abspath(__file__)))
            tar.close()
            os.remove(gz_path)

    def compute_score(self, gts, res):
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            input_data.append({
                "image_id": id,
                "test": hypo[0],
                "refs": ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile('w+', delete=False, dir=temp_dir, encoding='utf8')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile('w+', delete=False, dir=temp_dir, encoding='utf8')
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
                     '-cache', cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent'
                     ]

        try:
            from subprocess import DEVNULL  # Python 3.
        except ImportError:
            DEVNULL = open(os.devnull, 'wb')
        subprocess.check_call(spice_cmd,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              stdout=DEVNULL, stderr=DEVNULL)

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            imgId_to_scores[item['image_id']] = item['scores']
            spice_scores.append(float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {k: float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)
        return average_score, scores

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        """Returns the scores"""
        # gts = {k: [' '.join(w) for w in v] for k, v in enumerate(references)}
        # res = {k: [' '.join(v)] for k, v in enumerate(predictions)}
        gts = {k: v for k, v in enumerate(references)}
        res = {k: v for k, v in enumerate(predictions)}
        try:
            score, scores = self.compute_score(gts, res)
        except Exception as e:
            logging.getLogger().warning(e)
            score = None

        return {self.name: score}
