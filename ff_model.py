import torch
import torch.nn as nn
import torch.nn.functional as F


class FFModel(nn.Module):
    def __init__(self,
                 embeddings,
                 n_features=36,
                 hidden_size=200,
                 n_classes=3,
                 dropout_prob=0.5):
        super(FFModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0],
                                                  self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(
            torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size,
                                         self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)

    def embedding_lookup(self, t):
        z = self.pretrained_embeddings(t)
        p, q, r = z.size()
        x = z.view(p, q * r)

        return x

    def forward(self, t):
        x = self.embedding_lookup(t)
        z = F.relu(self.embed_to_hidden(x))
        m = nn.Dropout(self.dropout_prob)
        z_drop = m(z)
        logits = self.hidden_to_logits(z_drop)

        return logits
