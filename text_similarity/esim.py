import torch
from torch import nn

class ESIM(nn.Module):
    def __init__(self, 
                char_vocab_size, 
                char_dim, 
                char_hidden_size,
                hidden_size,
                max_word_len):
        super().__init__()

        # char的embedding层
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim)

        # char的双向lstm层
        self.char_lstm = nn.LSTM(input_size=char_dim,
                                hidden_size=char_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)

        # context的双向lstm层
        # inpuput_size计算方法
        # char_lstm的双向lstm输出为char_hidden_size*2
        # 经过attenion后拼接的输出为char_hidden_size*2*4
        self.context_lstm = nn.LSTM(input_size=char_hidden_size*8,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        # 最大池化层
        self.max_pool = nn.MaxPool2d((max_word_len, 1))

        # 全连接层
        self.dense1 = nn.Linear(char_hidden_size*2*4, hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)

        self.drop_out = nn.Dropout(0.2)

    def forward(self, char_p, char_q):
        # 先进入embedding层
        p_embedding = self.char_embedding(char_p)
        q_embedding = self.char_embedding(char_q)

        # 经过双向LSTM
        # 输出为out, (hidden_state, ceil_state)
        p_embedding, _ = self.char_lstm(p_embedding)
        q_embedding, _ = self.char_lstm(q_embedding)

        # dropout防止过拟合
        p_embedding = self.drop_out(p_embedding)
        q_embedding = self.drop_out(q_embedding)

        # attention处理
        # p_embedding:[batch_size, seq_len, char_hidden_size*2]
        # e: [batch_size, seq_len, seq_len]
        e = torch.matmul(p_embedding, torch.transpose(q_embedding, 1, 2))

        # 对权重矩阵e按行进行softmax,再与q_embedding相乘
        # p_hat:[batch_size, seq_len, char_hidden_size*2]
        p_hat = torch.matmul(torch.softmax(e, dim=2), q_embedding)

        # 对权重矩阵e按列进行softmax,再与p_embedding相乘
        q_hat = torch.matmul(torch.softmax(e, dim=1), p_embedding)        

        # 拼接维度变化
        # p_embedding:[batch_size, seq_len, char_hidden_size*2]
        # p_hat:[batch_size, seq_len, char_hidden_size*2]
        # p_cat:[batch_size, seq_len, char_hidden_size*2*4]
        p_cat = torch.cat([p_embedding, p_hat, p_embedding-p_hat, p_embedding*p_hat], dim=-1)
        q_cat = torch.cat([q_embedding, q_hat, q_embedding-q_hat, q_embedding*q_hat], dim=-1)

        # Inference Composition的双向LSTM
        p, _ = self.context_lstm(p_cat)
        q, _ = self.context_lstm(q_cat)

        # 最化池化, 降维把seq_len维度去掉
        p_max = self.max_pool(p).squeeze(dim=1)
        q_max = self.max_pool(q).squeeze(dim=1)

        # 平均池化，直接调用mean函数
        p_mean = torch.mean(p, dim=1)
        q_mean = torch.mean(q, dim=1)

        # 按seq_len这个维度进行拼接        
        x = torch.cat([p_max, p_mean, q_max, q_mean], dim=1)
        x = self.drop_out(x)

        # 经过两个全连接层，再经过sigmoid输出二分类
        x = torch.tanh(self.dense1(x))
        x = self.drop_out(x)
        x = torch.sigmoid(self.dense2(x))

        return x.squeeze(dim=-1)