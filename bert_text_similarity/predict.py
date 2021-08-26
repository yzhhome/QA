import os
import torch
from transformers import BertTokenizer

path = os.path.dirname(__file__)

# 获取QA项目根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bert_model_path = root_path + "/chinese-roberta-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = torch.load(path + '/bert_text_similarity.model')
model.eval()

# 对输入的句子进行预测
def bert_similarity_predict(sentence1, sentence2):

    encoded_input = tokenizer(list(sentence1), list(sentence2), 
        max_length=100, padding=True, 
        return_tensors='pt')

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    token_type_ids = encoded_input['token_type_ids']

    if torch.cuda.is_available():
        input_ids = input_ids.cuda().long()
        attention_mask = attention_mask.cuda().long()
        token_type_ids = token_type_ids.cuda().long() 

    output = model(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=-1)
    similarity = probabilities.max(dim=1)
    preds = similarity[0].cpu().data.numpy()

    return preds

if __name__ == '__main__':
    sentence1 = ["老年人怎么买保险？"]
    sentence2 = ["70岁以上老人怎么买保险？"]

    print('文本相似度：', bert_similarity_predict(sentence1, sentence2)[0])

