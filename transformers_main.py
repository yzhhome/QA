import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv('./data/qa_data.csv')
questions = df['question'].values
answers = df['answer'].values

bert_model_path = "./chinese-roberta-wwm-ext"

tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModel.from_pretrained(bert_model_path)

# 平均池化
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# 用transformers转换句向量
def bert_to_vec(sentence):
    print('bert encode start')
    if isinstance(sentence, np.ndarray):        
        encoded_input = tokenizer(list(sentence), padding=True, truncation=True, max_length=128, return_tensors='pt')
    else:
        encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')

    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print('bert encode finish')
    return sentence_embeddings.numpy()

# 将问题库中的所有问题转换为句向量
question_vec = bert_to_vec(questions)

# 计算余弦相似度
def cosine(a, b):
    # 矩阵的积除以矩阵模的积
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

def qa(text):
    vec = bert_to_vec(text)

    # 计算输入的问题和问题库中问题的相似度
    similarity = cosine(vec, question_vec).ravel()

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

