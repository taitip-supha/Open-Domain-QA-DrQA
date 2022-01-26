import json
from pythainlp.tokenize import word_tokenize,sent_tokenize


with open('./data/train.jsonl' ,encoding="utf8") as f:
    data = [json.loads(line) for line in f]

data[0][1]['context']
data[0][1]['question']
data[0][1]['answrs']

data[0][1]

sent_tokenize(data[0][1]['context'])

word_tokenize(data[0][1]['context'])