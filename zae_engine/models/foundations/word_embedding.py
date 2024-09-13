import torch
import torch.nn as nn
import gensim.downloader as api
import numpy as np


class Word2VecEmbedding(nn.Module):
    """
    A PyTorch module that uses pre-trained Word2Vec embeddings from gensim.

    Attributes
    ----------
    embedding : nn.Embedding
        PyTorch embedding layer initialized with Word2Vec pre-trained weights.

    Methods
    -------
    forward(x)
        Passes input tensor through the embedding layer.
    """

    def __init__(self):
        """
        Initializes the Word2VecEmbedding class by loading the pre-trained Word2Vec model
        and setting the weights to a PyTorch nn.Embedding layer.
        """
        super(Word2VecEmbedding, self).__init__()
        self.embedding = self._load_word2vec()

    def _load_word2vec(self):
        """
        Loads the pre-trained Word2Vec model from gensim, creates a PyTorch nn.Embedding layer,
        and initializes it with the Word2Vec weights.

        Returns
        -------
        nn.Embedding
            An nn.Embedding layer with weights set to the pre-trained Word2Vec embeddings.
        """
        model = api.load("word2vec-google-news-300")
        vocab_size = len(model.key_to_index)
        embedding_dim = model.vector_size

        embedding = nn.Embedding(vocab_size, embedding_dim)

        weights = np.zeros((vocab_size, embedding_dim))
        for i, word in enumerate(model.key_to_index):
            weights[i] = model[word]

        embedding.weight.data.copy_(torch.from_numpy(weights))
        return embedding

    def forward(self, x):
        """
        Passes the input tensor through the embedding layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing indices of words.

        Returns
        -------
        torch.Tensor
            The output tensor with embeddings for the input indices.
        """
        return self.embedding(x)


class FastTextEmbedding(nn.Module):
    """
    A PyTorch module that uses pre-trained FastText embeddings from gensim.

    Attributes
    ----------
    embedding : nn.Embedding
        PyTorch embedding layer initialized with FastText pre-trained weights.

    Methods
    -------
    forward(x)
        Passes input tensor through the embedding layer.
    """

    def __init__(self):
        """
        Initializes the FastTextEmbedding class by loading the pre-trained FastText model
        and setting the weights to a PyTorch nn.Embedding layer.
        """
        super(FastTextEmbedding, self).__init__()
        self.embedding = self._load_fasttext()

    def _load_fasttext(self):
        """
        Loads the pre-trained FastText model from gensim, creates a PyTorch nn.Embedding layer,
        and initializes it with the FastText weights.

        Returns
        -------
        nn.Embedding
            An nn.Embedding layer with weights set to the pre-trained FastText embeddings.
        """
        model = api.load("fasttext-wiki-news-subwords-300")
        vocab_size = len(model.key_to_index)
        embedding_dim = model.vector_size

        embedding = nn.Embedding(vocab_size, embedding_dim)

        weights = np.zeros((vocab_size, embedding_dim))
        for i, word in enumerate(model.key_to_index):
            weights[i] = model[word]

        embedding.weight.data.copy_(torch.from_numpy(weights))
        return embedding

    def forward(self, x):
        """
        Passes the input tensor through the embedding layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing indices of words.

        Returns
        -------
        torch.Tensor
            The output tensor with embeddings for the input indices.
        """
        return self.embedding(x)
