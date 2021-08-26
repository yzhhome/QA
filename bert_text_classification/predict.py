import os
import torch
from transformers import BertTokenizer

path = os.path.dirname(__file__)

# 获取QA项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bert_model_path = root_path + "/chinese-roberta-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = torch.load(path + '/bert_text_classification.model')
model.eval()

# 对输入的句子进行预测
def bert_classification_predict(s):
    encoded_input = tokenizer(s, max_length=10, padding=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.cuda().long()
        attention_mask = attention_mask.cuda().long()    

    output = model(input_ids, attention_mask)
    preds = torch.argmax(output.logits, dim=1).flatten()
    return preds.cpu().data.numpy()

if __name__ == '__main__':
    while True:
        s = input('输入句子：').strip()
        print('分类：', classification_predict(s)[0])