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

from .SeqEncoder import SeqEncoder
from .crf_tagger import CrfModel


@Model.register("joint_model")
class JointModel(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            encoder: SeqEncoder,
            text_field_embedder: TextFieldEmbedder,
            feedforward: FeedForward,
            # text_field_embedder: TextFieldEmbedder,
            tag_namespace: str = "tags",
            label_namespace: str = "labels",
            label_encoding: str = "BIO",
            calculate_span_f1: bool = True
            ):
        super(JointModel, self).__init__(vocab)
        self.feedforward = feedforward
        self.tagger = CrfModel(
            vocab = vocab,
            text_field_embedder = text_field_embedder,
            encoder = encoder,
            # feedforward = feedforward,
            label_encoding = label_encoding,
        )

        self.metrics = {
                "intent-accuracy": CategoricalAccuracy(),
                "intent-accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.loss = torch.nn.CrossEntropyLoss()
        
        
    def forward(self, sentence: Dict[str, torch.Tensor], labels: torch.Tensor, tags: torch.Tensor):
        output = self.tagger(sentence, tags)
        tagger_loss = output["loss"]
        h_n = output["h_n"]
        h_n = h_n.view(h_n.shape[0],-1)
        intent_loss = self.loss(self.feedforward(h_n), labels)
        output["loss"] = tagger_loss + intent_loss

        return output
        
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.tagger.get_metrics(reset)
        for metric_name, metric in self.metrics.items():
            metrics[metric_name] = metric.get_metric(reset)
        return metrics
        