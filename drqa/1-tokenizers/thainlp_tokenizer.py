import re
import json
import numpy as np

from pythainlp.tokenize import word_tokenize,sent_tokenize

#Read data
with open('./data/train.jsonl' ,encoding="utf8") as f:
    data = [json.loads(line) for line in f]

#Extract data to numpy array ['article_id', 'question_id', 'context', 'question', 'answers', 'title', 'created_by', 'created_on', 'is_pay']
context = np.empty([len(data[0]),2], dtype=object)
question = np.empty([len(data[0]),3], dtype=object)
answers = np.empty([len(data[0]),2], dtype=object)
for i in range(0,len(data[0])):
    len_c = len(data[0][i]['context'])
    len_q = len(data[0][i]['question'])
    len_a = len(data[0][i]['answers'])
    if (len_c>0 and len_q>0 and len_a>0) :
        context[i] = [data[0][i]['article_id'],data[0][i]['context']]
        question[i] = [data[0][i]['article_id'], data[0][i]['question_id'], data[0][i]['question']]
        answers[i] = [data[0][i]['question_id'], data[0][i]['answers']]
    else :  
        print(f"Data null{i} c:{len_c} q:{len_q} a:{len_a}")
print(f"#Context : {context.shape}\n#Question:{question.shape}\n#answers:{answers.shape}")


c_token = np.empty([len(data[0]),2], dtype=object)
c_token = [ re.sub('[ \t\r\n\v\f]', '', c) for c in np.unique(context)[0:10]]

np.array(word_tokenize(re.sub("[ (:,[\t\r\n\v\f)']", "", context[450][1])))