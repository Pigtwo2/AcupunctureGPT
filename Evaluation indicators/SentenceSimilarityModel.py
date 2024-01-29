import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for random number generators in PyTorch, NumPy, and Python.
    :param seed: The seed to be used for generating random numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SentenceSimilarityModel(nn.Module):
    """
    A model for computing sentence similarity using BERT embeddings and a Transformer encoder.
    """

    def __init__(self, bert_model_name: str = 'bert-base-chinese',seed=42) -> None:
        """
        Initializes the SentenceSimilarityModel.
        :param bert_model_name: Name or path of the BERT model to be used.
        """
        super(SentenceSimilarityModel, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        set_seed(seed)

        # Transformer layer (for Self Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert.config.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, sentA: str, sentB: str) -> torch.Tensor:
        """
        Forward pass for calculating similarity between two sentences.
        :param sentA: First sentence.
        :param sentB: Second sentence.
        :return: Tensor representing the cosine similarity between the sentence embeddings.
        """
        # Encoding sentences using BERT
        input_ids_A = self.bert_tokenizer(sentA, return_tensors="pt", padding=True, truncation=True, max_length=512)[
            'input_ids']
        input_ids_B = self.bert_tokenizer(sentB, return_tensors="pt", padding=True, truncation=True, max_length=512)[
            'input_ids']
        embeddings_A = self.bert(input_ids_A)[0]
        embeddings_B = self.bert(input_ids_B)[0]
        # Self Attention using Transformer
        SA_A = self.transformer_encoder(embeddings_A)
        SA_B = self.transformer_encoder(embeddings_B)
        avg_SA_A = torch.mean(SA_A, dim=1)
        avg_SA_B = torch.mean(SA_B, dim=1)
        similarity = F.cosine_similarity(avg_SA_A, avg_SA_B, dim=1, eps=1e-8).unsqueeze(-1)

        return similarity
