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


@Model.register("slot_filling")
class SlotFillingModel(Model):
    """
    Intent Detection Model
    """
    def __init__(self,
            encoder: Seq2SeqEncoder,
            text_field_embedder: TextFieldEmbedder,
            feedforward: FeedForward,
            vocab: Vocabulary,
            dropout: float,
            label_namespace: str,
            calculate_span_f1: bool,
            label_encoding: str = "BIO",
            initializer: InitializerApplicator = InitializerApplicator()):
        super(SlotFillingModel, self).__init__(vocab)
        self.tagger = CrfTagger(
            vocab = vocab,
            text_field_embedder = text_field_embedder,
            encoder = encoder,
            feedforward = feedforward,
            label_namespace = label_namespace,
            dropout = dropout,
            calculate_span_f1 = calculate_span_f1,
            label_encoding = label_encoding
        )
        initializer(self)

    
    def forward(self, sentence: Dict[str, torch.LongTensor], labels: torch.LongTensor, tags: torch.LongTensor):
        return self.tagger(sentence,tags)
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.tagger.get_metrics(reset)