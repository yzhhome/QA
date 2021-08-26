'''data_process.py'''
import pandas as pd
import jieba
import numpy as np
import os
from collections import defaultdict

path = os.path.dirname(__file__)

# 获取QA项目根目录
root_path =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 对句子进行分词
def tokenize(string):
    res = list(string)
    return res

# 构建词典并保存
def build_vocab(del_word_frequency):
    data = pd.read_csv(root_path + '/data/LCQMC.csv')
    segment1 = data['sentence1'].apply(tokenize)
    segment2 = data['sentence1'].apply(tokenize)

    word_frequency = defaultdict(int)

    # 统计每个词出现的次数 
    for row in segment1 + segment2:
        for i in row:
            word_frequency[i] += 1

    # 按词频降序排序
    word_sort = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

    f = open(path + '/vocab.txt', 'w', encoding='utf-8')
    f.write('[PAD]' + "\n" + '[UNK]' + "\n")

    # 词频小的不保存到词典中
    for d in word_sort:
        if d[1] > del_word_frequency:
            f.write(d[0] + "\n")
    f.close()

vocab = {}
# 转换字典索引
if os.path.exists(path + '/vocab.txt'):
    with open(path + '/vocab.txt', encoding='utf-8') as file:
        for line in file.readlines():
            vocab[line.strip()] = len(vocab)

# 转换句中的词在字典中的索引
def seq2index(seq):
    seg = tokenize(seq)
    seg_index = []
    for s in seg:
        seg_index.append(vocab.get(s, 1))
    return seg_index

# 统一长度
def padding_seq(X, max_len=10):    
    return np.array([
        np.concatenate([x, [0]*(max_len-len(x))]) if len(x) < max_len else x[:max_len] for x in X
    ])

# 获取文件的行数
def buff_count(file_name):
    with open(file_name, 'rb') as f:
        count = 0
        buf_size = 1024 * 1024
        buf = f.read(buf_size)
        while buf:
            count += buf.count(b'\n')
            buf = f.read(buf_size)
        return count

if __name__ == '__main__':
    build_vocab(5)