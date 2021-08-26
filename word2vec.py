'''
使用Word2Vec训练词向量
'''
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

input_file = './word2vec/wiki.txt'
model_file_name = './word2vec/wiki.model'

model = Word2Vec(sentences=LineSentence(input_file),
                vector_size=100, #词向量的维度
                window=5,
                min_count=5, #少于5的词舍去
                workers=multiprocessing.cpu_count(),
                sg=1, #使用skip-gram
                hs=0, #使用negative
                negative=5 #每次负采样5条
)

model.save(model_file_name)