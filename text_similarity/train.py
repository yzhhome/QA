import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from text_similarity.data_process import *
from text_similarity.esim import ESIM
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 获取QA项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(batch_size=32):
    # 读取同义句数据集
    df = pd.read_csv(root_path + '/data/LCQMC.csv')
    X = df[['sentence1', 'sentence2']]
    y = df['label']

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.05, shuffle=True, random_state=42)        
    
    X_train_p = X_train['sentence1']
    X_train_q = X_train['sentence2']

    X_valid_p = X_valid['sentence1']
    X_valid_q = X_valid['sentence2']    

    # 每个句子进行编码并统一长度
    X_train_p = padding_seq(X_train_p.apply(seq2index))
    X_train_q = padding_seq(X_train_q.apply(seq2index))

    X_valid_p = padding_seq(X_valid_p.apply(seq2index))
    X_valid_q = padding_seq(X_valid_q.apply(seq2index))

    y_train = np.array(y_train)    
    y_valid = np.array(y_valid)    

    train_data_set = TensorDataset(torch.from_numpy(X_train_p), 
                                    torch.from_numpy(X_train_q), 
                                    torch.from_numpy(y_train))

    train_data_loader = DataLoader(train_data_set, 
                                    batch_size=batch_size,
                                    shuffle=True)

    return train_data_loader, [X_valid_p, X_valid_q], y_valid    

def train():
    vocab_size = buff_count(os.path.dirname(__file__) + '/vocab.txt')
    model = ESIM(char_vocab_size=vocab_size, char_dim=100, 
                char_hidden_size=128, hidden_size=128,
                max_word_len=10)

    train_data_loader, X_valid, y_valid = load_data(batch_size=128)
    X_valid_p = X_valid[0]
    X_valid_q = X_valid[1]

    X_valid_p = torch.from_numpy(X_valid_p)
    X_valid_q = torch.from_numpy(X_valid_q)
    if torch.cuda.is_available():
        model = model.cuda()
        X_valid_p = X_valid_p.cuda().long()
        X_valid_q = X_valid_q.cuda().long()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    loss_func = nn.BCELoss()

    best_accuracy = 0.0

    for epoch in range(30):
        running_loss = 0.0
        for step, (b_x_p, b_x_q, b_y) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                b_x_p = b_x_p.cuda().long()
                b_x_q = b_x_q.cuda().long()
                b_y = b_y.cuda()
            output = model(b_x_p, b_x_q)
            loss = loss_func(output, b_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                valid_output = model(X_valid_p, X_valid_q)
                # 计算验证集上的精度
                y_pred = (valid_output.cpu().data.numpy() >0.5).astype(int)
                valid_accuracy = float((y_pred == y_valid).astype(int).sum()) / float(y_valid.size)

                # 保存验证精度最好的模型
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    torch.save(model, os.path.dirname(__file__) + '/esim.model')
                    print('save best valid accuracy: %.3f' %best_accuracy)
                print('epoch:', epoch, 
                    '| train loss: %.3f' %loss.cpu().data.numpy(), 
                    '| valid accuracy: %.3f' %valid_accuracy)

if __name__ == '__main__':    
    train()