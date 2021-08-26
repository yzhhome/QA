import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW
from sklearn.metrics import accuracy_score

# 获取QA项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bert_model_path = root_path + "/chinese-roberta-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(bert_model_path)

def load_data(batch_size=32):
    #读取分类数据
    df = pd.read_csv(root_path + '/data/classification.csv')
    X = df['sentence']
    y = df['label']

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.3, shuffle=True, random_state=True)

    # 对训练集和验证集数据进行编码
    encoded_input_train = tokenizer(list(X_train), max_length=10, padding=True, return_tensors='pt')
    encoded_input_valid = tokenizer(list(X_valid), max_length=10, padding=True, return_tensors='pt')

    # 获取tokenizer结果的input_ids和attention_mask
    train_inputs = encoded_input_train['input_ids']
    train_masks = encoded_input_train['attention_mask']

    valid_inputs = encoded_input_valid['input_ids']
    valid_masks = encoded_input_valid['attention_mask']

    # Series转换成numpy
    y_train = np.array(y_train)    
    y_valid = np.array(y_valid)       

    # numpy转换为tensor
    y_train = torch.tensor(y_train)    
    y_valid = torch.tensor(y_valid)    

    # 构建训练集的TensorDataset和DataLoader
    train_data_set = TensorDataset(train_inputs, train_masks, y_train)
    train_data_loader = DataLoader(train_data_set,
                                    batch_size=batch_size,
                                    shuffle=True)

    # 构建验证集的TensorDataset和DataLoader
    valid_data_set = TensorDataset(valid_inputs, valid_masks, y_valid)
    valid_data_loader = DataLoader(valid_data_set, 
                                    batch_size=batch_size,
                                    shuffle=True)                                                          

    return train_data_loader, valid_data_loader

def train():
    # 用Bert分类模型加载预训练语言模型
    model = BertForSequenceClassification.from_pretrained(bert_model_path)

    train_data_loader, valid_data_loader = load_data(batch_size=32)

    # 模型搬到GPU上
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    loss_func = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    for epoch in range(10):
        # 训练模式
        model.train()
        
        for step, (b_inputs, b_masks, b_y) in enumerate(train_data_loader):
            optimizer.zero_grad()

            # 数据搬到GPU上
            if torch.cuda.is_available():
                b_inputs = b_inputs.cuda().long()
                b_masks = b_masks.cuda().long()
                b_y = b_y.cuda()
            
            # 计算loss并进行反向传播
            output = model(b_inputs, b_masks)
            train_loss = loss_func(output.logits, b_y)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                # 计算训练集上的准确率
                preds = torch.argmax(output.logits, dim=1).flatten()
                train_accuracy = accuracy_score(b_y.cpu().data.numpy(), 
                    preds.cpu().data.numpy())

            if step % 10 == 0:
                print('epoch:', epoch, 
                    'step:', step, 
                    '| train loss: %.3f' %train_loss.cpu().data.numpy(), 
                    '| train accuracy: %.3f' %train_accuracy)
        # 评估模式
        model.eval()
        for step, (b_inputs, b_masks, b_y) in enumerate(valid_data_loader):            
            # 数据搬到GPU上
            if torch.cuda.is_available():
                b_inputs = b_inputs.cuda().long()
                b_masks = b_masks.cuda().long()
                b_y = b_y.cuda()
                        
            # 计算验证集上的准确率
            output = model(b_inputs, b_masks)
            valid_loss = loss_func(output.logits, b_y)
            preds = torch.argmax(output.logits, dim=1).flatten()
            valid_accuracy = accuracy_score(b_y.cpu().data.numpy(), 
                preds.cpu().data.numpy())

            if step % 10 == 0:
                print('epoch:', epoch, 
                    'step:', step, 
                    '| valid loss: %.3f' %valid_loss.cpu().data.numpy(), 
                    '| valid accuracy: %.3f' %valid_accuracy)  

            # 保存验证精度最好的模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model, os.path.dirname(__file__) 
                    + '/bert_text_classification.model')
                print('save best valid accuracy: %.3f' %best_accuracy)                    

        print('epoch:', epoch, 
            '| train loss: %.3f' %train_loss.cpu().data.numpy(), 
            '| train accuracy: %.3f' %train_accuracy,
            '| valid loss: %.3f' %valid_loss.cpu().data.numpy(), 
            '| valid accuracy: %.3f' %valid_accuracy)

if __name__ == '__main__':
    train()