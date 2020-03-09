#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :model.py
@说明    :
@时间    :2020/03/07 19:55:19
@作者    :吴京京
@版本    :0.0.1

论文名称：Attention-based recurrent neural network models for joint intent detection and slot filling
论文地址：https://arxiv.org/abs/1609.01454
'''

from typing import Iterable, List
import pickle
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Instance, Token
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField, LabelField, SequenceLabelField

from allennlp.models import Model


@Model.register("sss")
class AttentionEncoder(Model):
    def __init__(self,
            vocab: Vocabulary,
            ):
        super(AttentionEncoder,self).__init__(vocab)

@Model.register("attenton_based_rnn")
class AttentionRnn(Model):
    def __init__(self, 
            encoder: Model,
            vocab: Vocabulary,
            token_embedders: TextFieldEmbedder):
        super(AttentionRnn,self).__init__(vocab)
        self.vocab = vocab
        self.encoder = encoder
        self.token_embedders = token_embedders
        self.
