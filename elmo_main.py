import pandas as pd
import numpy as np
import jieba
from elmoformanylangs import Embedder

df = pd.read_csv('./data/qa_data.csv')
questions = df['question'].values
answers = df['answer'].values

elmo_model_path = './elmo/zhs.model'
elmo_model = Embedder(elmo_model_path)

def elmo2vec(sentence):
    '''
    output_layer参数.
    0 for the word encoder
    1 for the first LSTM hidden layer
    2 for the second LSTM hidden layer
    -1 for an average of 3 layers. (default)
    -2 for all 3 layers
    '''    
    if isinstance(sentence, str):
        segment = list(jieba.cut(sentence))
        # 用elmo转换词向量
        vec = elmo_model.sents2elmo([segment], output_layer=-1)
    elif isinstance(sentence, np.ndarray):
        segment = [jieba.cut(s) for s in sentence]
        # 用elmo转换词向量
        vec = elmo_model.sents2elmo(segment, output_layer=-1)

    # 句向量取均值
    return [np.mean(v, axis=0) for v in vec]

# 生成所有问题的句向量
question_vec = []
for q in questions:
    question_vec.extend(elmo2vec(q))

# 计算余弦相似度
def cosine(a, b):
    # 矩阵的积除以矩阵模的积
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

def qa(text):
    vec = elmo2vec(text)[0]

    # 计算输入的问题和问题库中问题的相似度
    similarity = cosine(vec, question_vec)

    # 取最大相似度
    max_similarity = max(similarity)
    print("最大相似度：", max_similarity)

    index = np.argmax(similarity)
    if max_similarity < 0.8:
        print(max_similarity)
        print('没有找到对应的问题，您问的是不是：', questions[index])
        return f"没有找到对应的问题，您问的是不是：{questions[index]}"
    
    print('最相似的问题：', questions[index])
    print('答案：', answers[index])
    return answers[index]

if __name__ == '__main__':
    for i in range(2):
        text = input('请输入您的问题：').strip()
        qa(text)