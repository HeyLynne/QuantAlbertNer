# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import sys
import logging

from processors import PADDREC, get_label_tag

logger = logging.getLogger(__name__)
try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

class NerAccuracyEvaluator(object):
    def __init__(self, label_lists, tag_type):
        self._label_lists = label_lists
        self.tag_type = tag_type
        return

    def _get_tag_fromlabel(self, labels):
        def is_begin_label(label):
            return label[0] == "B" # 0 is padding
        def is_pad_label(label):
            return label == PADDREC.PAD_TAG
        def is_label_matchtag(label, tag):
            return label[2:] == tag
        # notice that padding is 0, so we add 1 in each label position.
        # PAD means padding
        pred_label = pred_label = list(map(lambda x: get_label_tag(x, self._label_lists), labels))
        res = []
        now = 0
        start = 0
        tag = ""
        end = -1
        while now < len(pred_label):
            if is_pad_label(pred_label[now]):
                now += 1
                continue
            if is_begin_label(pred_label[now]):
                tag = pred_label[now][2:]
                start = now
                now += 1
                while (now < len(pred_label) \
                    and (not is_begin_label(pred_label[now])) \
                    and is_label_matchtag(pred_label[now], tag)):
                    now += 1
                res.append((start, now, tag))
                continue
            now += 1
        return res
    
    def _get_tag_fromlabel_nobi(self, labels):
        def is_pad_label(label):
            return label == PADDREC.PAD_TAG
        def is_other_matchbegin(begin, other):
            return begin == other
        # notice that padding is 0, so we add 1 in each label position.
        # PAD means padding
        pred_label = list(map(lambda x: get_label_tag(x, self._label_lists), labels))
        res = []
        start = -1
        tag = ""
        end = -1
        for i, p in enumerate(pred_label):
            if is_pad_label(p):
                continue
            if p != tag:
                if start != -1:
                    res.append((start, end, tag))
                start = i
                end = i
                tag = p
            elif tag != "" and is_other_matchbegin(tag, p):
                end = i
        return res

    def _evaluate_word(self, pred_labels, data_labels, with_BI = False):
        hit_num, pred_num, true_num = 0, 0, 0
        label_cnt = pred_labels.shape[0]
        for i in range(label_cnt):
            if with_BI:
                tag_pred = self._get_tag_fromlabel(pred_labels[i, :])
                tag_true = self._get_tag_fromlabel(data_labels[i, :])
            else:
                tag_pred = self._get_tag_fromlabel_nobi(pred_labels[i, :])
                tag_true = self._get_tag_fromlabel_nobi(data_labels[i, :])
            # print("pred", tag_pred, pred_labels[i, :], list(map(lambda x: get_label_tag(x, self._label_lists), pred_labels[i, :])))
            # print("true", tag_true, data_labels[i, :], list(map(lambda x: get_label_tag(x, self._label_lists), data_labels[i, :])))
            true_cnt = len(set(tag_true))
            pred_cnt = len(set(tag_pred))
            hit_cnt = len(set(tag_true) & set(tag_pred))
            hit_num += hit_cnt
            pred_num += pred_cnt
            true_num += true_cnt
        logger.info("**** Eval results ****")
        logger.info("Hit num %d, pred num %d, true num %d" % \
            (hit_num, pred_num, true_num))
        
        acc, recall, f1 = 0.0, 0.0, 0.0
        if pred_num != 0 and true_num != 0:
            acc = float(100) * (float(hit_num) / float(pred_num))
            recall = float(100) * (float(hit_num) / float(true_num))
            if acc != 0 and recall != 0:
                f1 = 2* acc * recall /(acc + recall)
        logger.info("ACC: %f, RECALL: %f, F1 %f" % \
            (acc, recall, f1))

        return {"acc": acc}
    
    def evaluate(self, pred_labels, data_labels, with_BI = False):
        if self.tag_type == "WORD":
            return self._evaluate_word(pred_labels, data_labels, with_BI)
        else:
            raise KeyError(self.tag_type) 
        return 

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "lcqmc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "ner":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
