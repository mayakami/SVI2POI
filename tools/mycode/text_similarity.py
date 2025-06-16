import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import Levenshtein


# 初始化分词器和模型
vocab_file = '/home/moss/streetview_segment/tools/google-bert/vocab.txt' #词表路径
tokenizer = BertTokenizer.from_pretrained(vocab_file)
model = BertModel.from_pretrained('/home/moss/streetview_segment/tools/google-bert/bert-base-chinese')

def get_bert_embeddings(text):
    # 对文本进行分词，限制最大长度为512个token
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # 获取BERT的输出
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state
    # 取[CLS]标记的输出作为句子的表示
    sentence_embedding = last_hidden_states[:, 0, :]
    return sentence_embedding

# 计算两个文本的语义相似度
def calculate_similarity(text1, text2):
    emb1 = get_bert_embeddings(text1) #(1,768)
    emb2 = get_bert_embeddings(text2)

    # 计算余弦相似度
    # 将emb1和emb2调整为(batch_size, 1, embedding_dim)，以便使用cosine_similarity
    similarity = cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(1), dim=2)
    return similarity.item()

# 主函数
def main(text1, text2):
    similarity = calculate_similarity(text1, text2)
    print(f"The semantic similarity between the texts is: {similarity}")

text1 = '汽车'
text2 = '美心MX'
main(text1, text2)
levenshtein_distance = Levenshtein.distance(text1, text2)
normalized_levenshtein_distance = levenshtein_distance / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 1
text_similarity_score = 1 - normalized_levenshtein_distance
print("\n" + str(text_similarity_score))