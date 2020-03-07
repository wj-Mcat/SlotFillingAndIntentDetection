#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :atis_dataset_reader.py
@说明    :
@时间    :2020/03/07 17:14:48
@作者    :吴京京
'''
from typing import Iterable, List
import pickle
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, SequenceLabelField


@DatasetReader.register("atis")
class ATISDdatasetReader(DatasetReader):
	def __init__(self, token_indexers: TokenIndexer = None, lazy: bool = False):
		super(ATISDdatasetReader, self).__init__(lazy=lazy)
		self.tokenizer = WordTokenizer()
		self.token_indexers = token_indexers if token_indexers else {"tokens": SingleIdTokenIndexer()}
	
	def _read(self, file_path: str) -> Iterable[Instance]:
		with open(file_path, 'rb') as stream:
			dataset, dicts = pickle.load(stream)
			
			query_idx2text = {idx:text for text, idx in dicts["token_ids"].items()}
			intent_idx2text = {idx:text for text,idx in dicts["intent_ids"].items()}
			entities_idx2text = {idx: text for text, idx in dicts["slot_ids"].items()}
			
			for index in range(dataset["query"].__len__()):
				query = dataset["query"][index]
				query = [Token(text = query_idx2text[idx]) for idx in query]
				
				entities = [entities_idx2text[idx] for idx in dataset["slot_labels"][index]]
				intent = intent_idx2text[dataset["intent_labels"][index][0]]
				
				yield self.text_to_instance(query,entities,intent)
			
		
	@overrides
	def text_to_instance(self, tokens: List[Token], entities: List[str] = None,intent:str=None) -> Instance:
		text_field = TextField(tokens, self.token_indexers)
		fields = {
			"entities": text_field,
		}
		if entities is not None and intent is not None:
			fields["labels"] = LabelField(intent,"labels")
			fields["tags"] = SequenceLabelField(entities,text_field,"tags")
		return Instance(fields)
