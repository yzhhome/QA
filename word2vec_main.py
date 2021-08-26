import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec

df = pd.read_csv('./data/qa_data.csv')
questions = df['question'].values
answers = df['answer'].values

model_path = './word2vec/wiki.model'
model = Word2Vec.load(model_path)

# 句子转化为句向量
def sen2vec(sentence):
    # 对句子进行分词
    segment = list(jieba.cut(sentence))
    vec = np.zeros(100)
    for s in segment:
        try:
            # 取出s对应的向量相加
            vec += model.wv[s]

        #出现oov问题，词不在词典中
        except: 
            print(f"{s} 不在词典中")
            pass
    
    # 采用加权平均求句向量
    vec /= len(segment)
    return vec

# 生成所有问题的句向量
question_vec = []
for q in questions:
    question_vec.append(sen2vec(q))

# 计算余弦相似度
def cosine(a, b):
    # 矩阵的积除以矩阵模的积
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

def qa(text):
    vec = sen2vec(text)

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