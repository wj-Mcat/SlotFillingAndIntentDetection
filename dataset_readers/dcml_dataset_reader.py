#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :dcml_dataset_reader.py
@说明    :
@时间    :2020/03/07 19:29:29
@作者    :吴京京
@版本    :0.0.1
'''
from typing import Iterable, List
import pickle
import os
import jieba
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, SequenceLabelField


@DatasetReader.register("dcml")
class DCMLDatasetReader(DatasetReader):

    def _read(self, file_name: str) -> Iterable[Instance]:
        file_dir, key = tuple(file_name.split('|'))
        vocab_dic = {}
        with open(os.path.join(file_dir, "vocab.query")) as f:
            lines = f.readlines()
            vocab_dic = {index: word for index, word in lines}
        
        query = open(os.path.join(file_dir, f"model_data_"))
