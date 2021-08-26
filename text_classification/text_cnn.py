import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, vocab_len, 
                embedding_size, 
                max_length=10, 
                kernel_sizes=[3,4,5],
                out_channels=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_size)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_sizes[0], embedding_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_sizes[1], embedding_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_sizes[2], embedding_size))

        self.max_pool1 = nn.MaxPool1d(kernel_size=max_length - kernel_sizes[0] + 1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=max_length - kernel_sizes[1] + 1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=max_length - kernel_sizes[2] + 1)

        self.drop_out = nn.Dropout(0.2)
        self.dense = nn.Linear(3*out_channels, 1)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.unsqueeze(1)

        conv1_out = self.conv1(embedding).squeeze(-1)
        conv2_out = self.conv2(embedding).squeeze(-1)
        conv3_out = self.conv3(embedding).squeeze(-1)

        out1 = self.max_pool1(conv1_out)
        out2 = self.max_pool2(conv2_out)
        out3 = self.max_pool3(conv3_out)

        out = torch.cat([out1, out2, out3], dim=1).squeeze(-1)
        out = self.drop_out(out)
        out = self.dense(out)
        out = torch.sigmoid(out).squeeze(-1)

        return out