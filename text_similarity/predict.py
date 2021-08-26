import os
import torch
from text_similarity.data_process import seq2index, padding_seq

path = os.path.dirname(__file__)

model = torch.load(path + '/esim.model')
model.eval()

# 对输入的句子进行预测
def similarity_predict(p, q):
    p = [seq2index(i) for i in p]
    q = [seq2index(i) for i in q]
    p = torch.from_numpy(padding_seq(p)).cuda().long()
    q = torch.from_numpy(padding_seq(q)).cuda().long()
    out = model(p, q)
    return out.cpu().data.numpy()

if __name__ == '__main__':
    similarity = similarity_predict(
        ['网上卖的烟有真的吗？'], 
        ['网上帮你刷钻是真的吗'])
    print(similarity[0])