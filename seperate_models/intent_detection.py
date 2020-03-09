from typing import Iterable, List, Dict
import pickle
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
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


@Model.register("intent_detection_model")
class IntentDetectionModel(Model):
    """
    Intent Detection Model
    """
    def __init__(self,
            encoder: Seq2VecEncoder,
            feed_forward: FeedForward,
            text_field_embedder: TextFieldEmbedder,
            vocab: Vocabulary,
            initializer: InitializerApplicator = InitializerApplicator()):
        super(IntentDetectionModel, self).__init__(vocab)
        self.encoder = encoder
        self.feed_forward = feed_forward
        self.text_field_embedder = text_field_embedder

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    
    def forward(self, sentence: Dict[str, torch.LongTensor], labels: torch.LongTensor, tags: torch.LongTensor):
        """

        """
        inp = self.text_field_embedder(sentence)
        mask = util.get_text_field_mask(sentence)
        inp = self.encoder(inp, mask)
        inp = self.feed_forward(inp)
        output = {
            "logits": inp
        }
        if labels is not None:
            loss = self.loss(inp, labels)
            output["loss"] = loss
            for metric in self.metrics.values():
                metric(inp, labels)
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}