import os
import jieba
import numpy as np
import pandas as pd
import torch
from elmoformanylangs import Embedder
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from bert_text_classification.predict import bert_classification_predict
from chitchat.interact import chitchat
from text_classification.predict import classification_predict
from text_similarity.predict import similarity_predict
from bert_text_similarity.predict import bert_similarity_predict

# 获取QA项目根目录
root_path = os.path.dirname(__file__)

df = pd.read_csv(root_path + '/data/qa_data.csv')
questions = df['question'].values
answers = df['answer'].values

# 句子转化为句向量
def sen2vec(model, sentence):
    # 对句子进行分词
    segment = list(jieba.cut(sentence))
    vec = np.zeros(100)
    for s in segment:
        try:
            # 取出s对应的向量相加
            vec += model.wv[s]

        #出现oov问题，词不在词典中
        except: 
            pass
    
    # 采用加权平均求句向量
    vec /= len(segment)
    return vec


def elmo2vec(model, sentence):
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
        vec = model.sents2elmo([segment], output_layer=-1)
    elif isinstance(sentence, np.ndarray):
        segment = [jieba.cut(s) for s in sentence]
        # 用elmo转换词向量
        vec = model.sents2elmo(segment, output_layer=-1)     

    # 句向量取均值
    return [np.mean(v, axis=0) for v in vec]

# 平均池化
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# 用transformers转换句向量
def bert_to_vec(model, tokenizer, sentence):
    print('bert encode start')
    if isinstance(sentence, np.ndarray):        
        encoded_input = tokenizer(list(sentence), padding=True, 
            truncation=True, max_length=128, return_tensors='pt')
    else:
        encoded_input = tokenizer(sentence, padding=True, 
            truncation=True, max_length=128, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print('bert encode finish')    
    return sentence_embeddings.numpy()

# 用sentence_transformers转换句向量
def sentence_to_vec(model, sentence):   
    if isinstance(sentence, np.ndarray):  
        embedding = model.encode(sentence, batch_size=64, show_progress_bar=True, device='cuda:0')
    else:
        embedding = model.encode(sentence)     
    return embedding

# 计算余弦相似度
def cosine(a, b):
    # 矩阵的积除以矩阵模的积
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

if __name__ == '__main__':
    while True:
        text = input('请输入您的问题：').strip()

        # 判断是封闭域还是闲聊问题
        # TextCNN和Bert fine-tune 作对比
        prob_cnn = round(classification_predict(text)[0], 3)
        print("TextCNN 预测是闲聊的概率为：", prob_cnn)

        prob_bert = round(float(bert_classification_predict(text)[0]), 3)
        print("Bert 预测是闲聊的概率为：", prob_bert)

        if (prob_cnn > 0.5) or (prob_bert > 0.5):
            print("当前输入的问题为闲聊")
            print("闲聊回答：", chitchat(text))
            continue
        else:
            print("当前输入的问题为封闭域问题")

        while True:
            v = int(input("请选择句向量编码方式：\n" +
                    "1. Word2Vec \n" +
                    "2. ElMo \n" +
                    "3. Bert \n" +
                    "4. sentence-transformers \n",).strip())

            if v != 1 and v != 2 and v != 3 and v != 4:
                print("输入的句向量编码方式错误，请重新输入")
                continue
            else:
                break

        print("正在将问题库中的所有问题转换为句向量...")            

        # 文本表示，转换为句向量
        vec = None

        # Word2Vec
        if v == 1:
            v_str = "Word2Vec"
            model_path = root_path + '/word2vec/wiki.model'
            model = Word2Vec.load(model_path)

            # 生成所有问题的句向量
            question_vec = []
            for q in questions:
                question_vec.append(sen2vec(model, q))

            # 生成当前输入问题的句向量
            vec = sen2vec(model, text)
        # Elmo
        elif v == 2:
            v_str = "ELMo"
            model_path = root_path + '/elmo/zhs.model'
            model = Embedder(model_path)

            # 生成所有问题的句向量
            question_vec = []
            for q in questions:
                question_vec.extend(elmo2vec(model, q))  

            # 生成当前输入问题的句向量 
            vec = elmo2vec(model, text)[0]
        # Bert
        elif v == 3:     
            v_str = "Bert"
            model_path = root_path + "/chinese-roberta-wwm-ext"

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)  

             # 生成所有问题的句向量
            question_vec = bert_to_vec(model, tokenizer, questions)   

            # 生成当前输入问题的句向量          
            vec = bert_to_vec(model, tokenizer, text)
        # sentence transformers
        elif v == 4:  
            v_str = "sentence-transformers"       
            model_path =  root_path + "/paraphrase-multilingual-MiniLM-L12-v2"
            model = SentenceTransformer(model_path, device='cuda:0')  

            # 将所有问题库中问题转换为句向量
            question_vec = sentence_to_vec(model, questions)   

            # 生成当前输入问题的句向量        
            vec = sentence_to_vec(model, text)

        # 计算输入的问题和问题库中问题的相似度
        similarity = cosine(vec, question_vec)

        # Bert的是二维数组，需要拉成一维数组
        if v == 3:
            similarity = similarity.ravel()

        # 取最大相似度
        max_similarity = max(similarity)     

        print(v_str + " 最大相似度：", max_similarity)
        index = np.argmax(similarity)
        if max_similarity < 0.8:
            print('没有找到对应的问题，您想问的是不是：', questions[index])
            continue 

        print(v_str + ' 最相似的问题：', questions[index])
        print(v_str + ' 答案：', answers[index][0:100], "...")        

        top_10_similarity = np.argsort(-similarity)[0:10]
        top_10_question = questions[top_10_similarity]
        esim_similarity = similarity_predict([text] * 10, top_10_question)
        bert_similarity = bert_similarity_predict([text] * 10, top_10_question)
        index_dic = {}
        print(v_str + ' 和 ESIM Bert Top10 候选集：')
        df_top_10 = pd.DataFrame(columns=['question', v_str, 'ESIM', 'Bert'])
        pd.set_option('colheader_justify', 'center')

        for i, index in enumerate(top_10_similarity):
            df_top_10.loc[i] = [top_10_question[i], similarity[index], 
                esim_similarity[i], bert_similarity[i]]
            index_dic[i] = index

        print(df_top_10)

        esim_index = np.argsort(-esim_similarity)[0]
        print('ESIM最相似的问题：第' + str(esim_index) + '个', 
            questions[index_dic[esim_index]])
        print('ESIM答案:', answers[index_dic[esim_index]][0:100], "...")

        bert_index = np.argsort(-bert_similarity)[0]
        print('Bert最相似的问题：第' + str(bert_index) + '个', 
            questions[index_dic[bert_index]])
        print('Bert答案:', answers[index_dic[bert_index]][0:100], "...")        