import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv('./data/qa_data.csv')
questions = df['question'].values
answers = df['answer'].values

sentence_model_path = "./paraphrase-multilingual-MiniLM-L12-v2"
sentence_model = SentenceTransformer(sentence_model_path, device='cuda:0')

# 用sentence_transformers转换句向量
def sentence_to_vec(sentence):   
    if isinstance(sentence, np.ndarray):  
        embedding = sentence_model.encode(sentence, batch_size=64, show_progress_bar=True, device='cuda:0')
    else:
        embedding = sentence_model.encode(sentence)
    return embedding

# 将所有问题库中问题转换为句向量
question_vec = sentence_to_vec(questions)

# 计算余弦相似度
def cosine(a, b):
    # 矩阵的积除以矩阵模的积
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

def qa(text):
    vec = sentence_to_vec(text)

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