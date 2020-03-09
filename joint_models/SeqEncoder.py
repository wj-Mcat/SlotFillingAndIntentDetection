from typing import Iterable, List, Dict
import pickle
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Instance, Token
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.models import CrfTagger
from allennlp.models import Model
from allennlp.nn import util
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
import torch
from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from allennlp.modules.attention import LinearAttention


@Model.register("seq_encoder")
class SeqEncoder(Model):

	def __init__(self, vocab: Vocabulary,
	             num_layers: int = 1,
	             input_dim: int = 100,
	             hidden_dim: int = 100,
	             bidirectional: bool = True,
				 batch_size: int = 100,
				 with_attention: bool = True,
	             dropout: float = 0.2):
		super(SeqEncoder, self).__init__(vocab)
		self.rnn = torch.nn.LSTM(
			num_layers = num_layers,
			input_size= input_dim,
			hidden_size= hidden_dim,
			bidirectional= bidirectional,
			dropout= dropout,
			batch_first= True
		)
		self.num_bidirectional = 2 if bidirectional else 1
		self.hidden_dim = hidden_dim
		self.hidden = None
		self.with_attention = with_attention
		if with_attention:
			self.attention = LinearAttention(
				tensor_1_dim = hidden_dim * self.num_bidirectional,
				tensor_2_dim = hidden_dim * self.num_bidirectional)

		
	def forward(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		inp = inp * mask.unsqueeze_(-1)
		output , self.hidden = self.rnn(inp, self.hidden)
		
		# 如果有attention，则把attention的分数传递过去
		if self.with_attention:
			attention = self.attention(output, output)
			return attention, self.hidden
		return  output, self.hidden
	
	def get_output_dim(self):
		return self.hidden_dim * self.num_bidirectional