import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """ Feedforward neural network with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.
    """

    def __init__(self,
                 embeddings,
                 n_features=36,
                 hidden_size=72,
                 n_classes=3,
                 dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(CNNModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0],
                                                  self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(
            torch.tensor(embeddings))

        self.conv1 = nn.Conv1d(self.embed_size, self.hidden_size, 4)
        self.conv2 = nn.Conv1d(self.hidden_size, 144, 4)
        self.pool = nn.MaxPool1d(2)
        self.linear1 = nn.Linear(864, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 3)

        # self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size,
        #                                  self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(36, self.n_classes)

    def embedding_lookup(self, t):
        """ Utilize `self.pretrained_embeddings` to map input `t` from input tokens (integers)
            to embedding vectors.

        @param t (Tensor): input tensor of tokens (batch_size, n_features)

        @return x (Tensor): tensor of embeddings for words represented in t
                                (batch_size, n_features * embed_size)
        """
        z = self.pretrained_embeddings(t)
        p, q, r = z.size()
        x = z.view(p, q * r)

        return z

    def forward(self, t):
        """ Run the model forward.

        @param t (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """

        x = self.embedding_lookup(t)
        x = torch.permute(x, [0, 2, 1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
