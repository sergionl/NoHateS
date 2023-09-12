
import torch
import torch.nn as nn

class CNNForNLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, filter_sizes):
        super(CNNForNLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, filter_size)
            for filter_size in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x, _):
        embedded = self.embedding(x)  # x: (batch_size, sequence_length)
        embedded = embedded.permute(0, 2, 1)  # embedded: (batch_size, embedding_dim, sequence_length)
        feature_maps = []
        for conv in self.convs:
            feature_map = torch.relu(conv(embedded))  # feature_map: (batch_size, num_filters, H)
            pooled = torch.max(feature_map, dim=2)[0]  # pooled: (batch_size, num_filters)
            feature_maps.append(pooled)
        combined = torch.cat(feature_maps, dim=1)  # combined: (batch_size, len(filter_sizes) * num_filters)
        combined = self.dropout(combined)
        logits = self.fc(combined)  # logits: (batch_size, num_classes)
        return logits
