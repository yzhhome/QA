import sys
import os
from numpy.core.defchararray import mod
import torch
from torch import nn
import pandas as pd
from text_classification.text_cnn import TextCNN
from text_classification.data_process import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 获取QA项目根目录
root_path =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(batch_size=64):
    #读取分类数据
    df = pd.read_csv(root_path + '/data/classification.csv')
    X = df['sentence']
    y = df['label']

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.3, shuffle=True, random_state=True)

    # 每个句子进行编码并统一长度
    X_train = padding_seq(X_train.apply(seq2index))
    X_valid = padding_seq(X_valid.apply(seq2index))

    y_train = np.array(y_train)    
    y_valid = np.array(y_valid)    

    train_data_set = TensorDataset(torch.from_numpy(X_train), 
                                    torch.from_numpy(y_train))

    train_data_loader = DataLoader(train_data_set, 
                                    batch_size=batch_size,
                                    shuffle=True)

    return train_data_loader, X_valid, y_valid

def train():
    vocab_size = buff_count(os.path.dirname(__file__) + '/vocab.txt')
    model = TextCNN(vocab_len=vocab_size, embedding_size=100, out_channels=100)

    train_data_loader, X_valid, y_valid = load_data(batch_size=64)

    X_valid = torch.from_numpy(X_valid)
    if torch.cuda.is_available():
        model = model.cuda()
        X_valid = X_valid.cuda().long()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.BCELoss()

    best_accuracy = 0.0

    for epoch in range(100):
        for step, (b_x, b_y) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                b_y = b_y.cuda()
            output = model(b_x)
            loss = loss_func(output, b_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                valid_output = model(X_valid)
                # 计算验证集上的精度
                y_pred = (valid_output.cpu().data.numpy() >0.5).astype(int)
                valid_accuracy = float((y_pred == y_valid).astype(int).sum()) / float(y_valid.size)

                # 保存验证精度最好的模型
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    torch.save(model, os.path.dirname(__file__) + '/text_cnn.model')
                    print('save best valid accuracy: %.3f' %best_accuracy)
                print('epoch:', epoch, '| train loss: %.3f' %loss.cpu().data.numpy(), 
                        '| valid accuracy: %.3f' %valid_accuracy)

if __name__ == '__main__':
    train()          