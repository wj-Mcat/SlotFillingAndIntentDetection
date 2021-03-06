#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :dcml_dataset_reader.py
@说明    :
@时间    :2020/03/07 19:29:29
@作者    :吴京京
@版本    :0.0.1
'''
from typing import Iterable, List, Dict
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

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None):
        self.tokenizer = WordTokenizer()
        self.token_indexers = token_indexers if token_indexers is None else {"token": SingleIdTokenIndexer()}


    def _read(self, file_fix: str) -> Iterable[Instance]:
        file_dir, key = tuple(file_fix.split('|'))

        # load vocabulary
        vocab_dic = {}
        with open(os.path.join(file_dir, "vocab.query"), "r+", encoding="utf-8") as f:
            lines = f.readlines()
            vocab_dic = { index: word for index, word in lines }
        
        # load query
        

        queries = open(os.path.join(file_dir, "seq.in"), "r+").readlines()
        labels = open(os.path.join(file_dir, "label"), "r+").readlines()
        intents = open(os.path.join(file_dir, "seq.out"), "r+").readlines()

        for index, query in enumerate(queries):
            tokens =[Token(token) for token in self.tokenizer.tokenize(query)]
            yield self.text_to_instance(tokens, labels=labels[index], intent=intents[index])
    
    @overrides
    def text_to_instance(self, tokens: List[Token], labels: List[str] = None, intent: str = None) -> Instance:
        sentence_field = TextField(tokens, token_indexers = self.token_indexers)
        fields = {
            "sentence": sentence_field
        }
        if labels is None:
            fields["labels"] = SequenceLabelField(labels, sequence_field = sentence_field)
            fields["intent"] = LabelField(intent)
        return Instance(fields)
