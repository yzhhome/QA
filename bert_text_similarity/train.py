import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW

# 获取QA项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bert_model_path = root_path + "/chinese-roberta-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(bert_model_path)

# 截短和padding操作
def truncate_and_pad(token_seq_1, token_seq_2, max_seq_len=100):
    if len(token_seq_1) > max_seq_len:
        token_seq_1 = token_seq_1[0:max_seq_len]

    if len(token_seq_2) > max_seq_len:
        token_seq_1 = token_seq_2[0:max_seq_len]      

    # 用[SET]分割两个句子
    seq = ['[CLS]'] + token_seq_1 + ['[SEP]'] + token_seq_2 + ['[SEP]']

    # 0为第二个句子1为第二个句子
    seq_segment = [0] * (len(token_seq_1) + 2) + [1] * (len(token_seq_2) + 1)

    # 转换为ID
    seq = tokenizer.convert_tokens_to_ids(seq)

    # 需要补0的数量
    padding = [0] * (max_seq_len - len(seq))

    # 有效位为1，其他为0
    seq_mask = [1] * len(seq) + padding

    # 超过两个句子长度的补0
    seq = seq + padding
    seq_segment = seq_segment + padding      

    return seq, seq_mask, seq_segment

"""
用[CLS][SEP]方式分割句子
用tokenize方式分词
手动构建input_ids, attenion_mask, token_type_ids
"""
def load_data_2(batch_size=32):
    #读取分类数据
    df = pd.read_csv(root_path + '/data/LCQMC.csv')
    X = df[['sentence1', 'sentence2']]
    y = df['label']

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.1, shuffle=True, random_state=42)

    train_sentences_1 = X_train['sentence1']
    train_sentences_2 = X_train['sentence2']

    valid_sentences_1 = X_valid['sentence1']
    valid_sentences_2 = X_valid['sentence2']

    # 对训练集句子进行分词
    train_tokens_seq_1 = list(map(tokenizer.tokenize, train_sentences_1))
    train_tokens_seq_2 = list(map(tokenizer.tokenize, train_sentences_2))

    # 对训练集句子进行分词
    valid_tokens_seq_1 = list(map(tokenizer.tokenize, valid_sentences_1))
    valid_tokens_seq_2 = list(map(tokenizer.tokenize, valid_sentences_2))   

    # 截短和Padding操作
    train_pad = list(map(truncate_and_pad, train_tokens_seq_1, train_tokens_seq_2))
    valid_pad = list(map(truncate_and_pad, valid_tokens_seq_1, valid_tokens_seq_2)) 

    # 取出input_ids, attention_mask, token_type_ids
    train_seqs = torch.tensor([i[0] for i in train_pad])
    train_seq_masks = torch.tensor([i[1] for i in train_pad])
    train_seq_segments = torch.tensor([i[2] for i in train_pad])

    valid_seqs = torch.tensor([i[0] for i in valid_pad])
    valid_seq_masks = torch.tensor([i[1] for i in valid_pad]) 
    valid_seq_segments = torch.tensor([i[2] for i in valid_pad])

    # Series转换成numpy
    y_train = np.array(y_train)    
    y_valid = np.array(y_valid)       

    # numpy转换为tensor
    y_train = torch.tensor(y_train)    
    y_valid = torch.tensor(y_valid)

    # 构建训练集的TensorDataset和DataLoader
    train_data_set = TensorDataset(train_seqs, train_seq_masks, train_seq_segments, y_train)
    train_data_loader = DataLoader(train_data_set,
                                    batch_size=batch_size,
                                    shuffle=True)

    # 构建验证集的TensorDataset和DataLoader
    valid_data_set = TensorDataset(valid_seqs, valid_seq_masks, valid_seq_segments, y_valid)
    valid_data_loader = DataLoader(valid_data_set, 
                                    batch_size=batch_size,
                                    shuffle=True)                                                          

    return train_data_loader, valid_data_loader  

"""
直接用tokenizer方式编码
生成input_ids, attention_mask, token_type_ids
"""
def load_data(batch_size=32):
    #读取分类数据
    df = pd.read_csv(root_path + '/data/LCQMC.csv')
    X = df[['sentence1', 'sentence2']]
    y = df['label']

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
        test_size=0.1, shuffle=True, random_state=42)

    X_train_p = X_train['sentence1']
    X_train_q = X_train['sentence2']

    X_valid_p = X_valid['sentence1']
    X_valid_q = X_valid['sentence2']

    # 对训练集和验证集数据进行编码
    encoded_train = tokenizer(list(X_train_p), list(X_train_q), 
        max_length=100, padding=True, return_tensors='pt')

    encoded_valid = tokenizer(list(X_valid_p), list(X_valid_q),     
        max_length=100, padding=True, return_tensors='pt')

    # 获取tokenizer结果的input_ids和attention_mask
    train_inputs = encoded_train['input_ids']
    train_masks = encoded_train['attention_mask']
    train_segments = encoded_train['token_type_ids']

    valid_inputs = encoded_valid['input_ids']
    valid_masks = encoded_valid['attention_mask']
    valid_segments = encoded_valid['token_type_ids']

    # Series转换成numpy
    y_train = np.array(y_train)    
    y_valid = np.array(y_valid)       

    # numpy转换为tensor
    y_train = torch.tensor(y_train)    
    y_valid = torch.tensor(y_valid)

    # 构建训练集的TensorDataset和DataLoader
    train_data_set = TensorDataset(train_inputs, train_masks, train_segments, y_train)
    train_data_loader = DataLoader(train_data_set,
                                    batch_size=batch_size,
                                    shuffle=True)

    # 构建验证集的TensorDataset和DataLoader
    valid_data_set = TensorDataset(valid_inputs, valid_masks, valid_segments, y_valid)
    valid_data_loader = DataLoader(valid_data_set, 
                                    batch_size=batch_size,
                                    shuffle=True)                                                          

    return train_data_loader, valid_data_loader

def train():
    # 用Bert分类模型加载预训练语言模型
    model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)

    train_data_loader, valid_data_loader = load_data(batch_size=64)

    # 模型搬到GPU上
    if torch.cuda.is_available():
        model = model.cuda()

    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # no_decy里面的不进行weight_decay
    optimizer_grouped_parameters = [{
        'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':0.01},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':0.0}]        

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-05)
    best_accuracy = 0.0      

    for epoch in range(10):
        # 训练模式
        model.train()

        # 每个epoch总的loss
        train_loss = 0.0

        # 每个epoch总计预测正确数量
        train_correct_preds = 0  

        # 每个epoch总计处理过多少个问题
        train_question_count = 0 

        for step, (b_inputs, b_masks, b_segments, b_y) in enumerate(train_data_loader):            
            optimizer.zero_grad()

             # 数据搬到GPU上
            if torch.cuda.is_available():
                b_inputs = b_inputs.cuda().long()
                b_masks = b_masks.cuda().long()
                b_segments = b_segments.cuda().long()
                b_y = b_y.cuda()      

            output = model(input_ids=b_inputs, 
                                attention_mask=b_masks, 
                                token_type_ids=b_segments, 
                                labels=b_y)
            # 取出loss, logits
            loss, logits = output.loss, output.logits

            # 对logits进算softmax
            probabilities = torch.softmax(logits, dim=-1)
            loss.backward()
            # 进行梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=10.0)   
            optimizer.step()

            # 每个step的loss相加，便于后面计算每个epoch的loss
            train_loss += loss.item()

            # 取出最大预测值
            _, out_classes = probabilities.max(dim=1)

            # 计算每个step预测正确的数量
            correct = (out_classes == b_y).sum()

            # 每个epoch预测正确的数量累加
            train_correct_preds += correct.item()

            # 每个epoch预测的总量累加
            train_question_count += len(b_y)

            # 每个epoch预测正确的数量 / 每个epoch预测的总数量
            train_accuracy = train_correct_preds / train_question_count

            if step % 5 == 0:
                print('epoch:', epoch, 
                    'train step:', step, 
                    '| train loss: %.3f' %loss.item(), 
                    '| train accuracy: %.3f' %train_accuracy) 
            
        # 评估模式
        model.eval()

        valid_loss = 0.0
        valid_correct_preds = 0    
        valid_question_count = 0   

        for step, (b_inputs, b_masks, b_segments, b_y) in enumerate(valid_data_loader):    
 
            # 数据搬到GPU上
            if torch.cuda.is_available():
                b_inputs = b_inputs.cuda().long()
                b_masks = b_masks.cuda().long()
                b_segments = b_segments.cuda().long()
                b_y = b_y.cuda()    

            output = model(input_ids=b_inputs, 
                                attention_mask=b_masks, 
                                token_type_ids=b_segments, 
                                labels=b_y)
            loss, logits = output.loss, output.logits
            probabilities = torch.softmax(logits, dim=-1)

            valid_loss += loss.item()
            _, out_classes = probabilities.max(dim=1)
            correct = (out_classes == b_y).sum()
            valid_correct_preds += correct.item()     
            valid_question_count += len(b_y)       
            valid_accuracy = valid_correct_preds / valid_question_count

            if step % 5 == 0:
                print('epoch:', epoch, 
                    'valid step:', step, 
                    '| valid loss: %.3f' %loss.item(), 
                    '| valid accuracy: %.3f' %valid_accuracy) 

            # 保存验证精度最好的模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model, os.path.dirname(__file__) 
                    + '/bert_text_similarity.model')
                print('save best valid accuracy: %.3f' %best_accuracy)     

        # 计算每个epoch的train loss, train accuracy
        epoch_train_loss = train_loss / len(train_data_loader)
        epoch_train_accuracy = train_correct_preds / len(train_data_loader.dataset)

        # 计算每个epoch的valid loss, valid_accuracy
        epoch_valid_loss = valid_loss / len(valid_data_loader)
        epoch_valid_accuracy = valid_correct_preds / len(valid_data_loader.dataset)  
  
        print('epoch:', epoch, 
            '| train loss: %.3f' %epoch_train_loss, 
            '| train accuracy: %.3f' %epoch_train_accuracy,
            '| valid loss: %.3f' %epoch_valid_loss, 
            '| valid accuracy: %.3f' %epoch_valid_accuracy)

if __name__ == '__main__':
    train()