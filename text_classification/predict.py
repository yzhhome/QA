import os
import torch
from text_classification.data_process import seq2index, padding_seq

path = os.path.dirname(__file__)

model = torch.load(path + '/text_cnn.model')
model.eval()

# 对输入的句子进行预测
def classification_predict(s):
    s = seq2index(s)
    s = padding_seq([s])

    x = torch.from_numpy(s).cuda().long()
    out = model(x)
    return out.cpu().data.numpy()

if __name__ == '__main__':
    while True:
        s = input('输入句子：').strip()
        print('分类：', classification_predict(s)[0])
