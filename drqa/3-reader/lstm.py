import re
import json
import numpy as np

# this snippet of code is for data extranction from json file.
contexts = []
questions = []
answers_text = []
answers_start = []
title = []
# getting train data and dev data into csv file

# this snippet of code is for data extranction from json file.
contexts = []
questions = []
answers_text = []
answers_start = []
title = []
for i in range(train.shape[0]):
    topic = train.iloc[i,0]['paragraphs']
    title_ = train.iloc[i,0]['title']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            questions.append(q_a['question'])
            if len(q_a['answers'])>0 :
                answers_start.append(q_a['answers'][0]['answer_start']) 
                answers_text.append(q_a['answers'][0]['text'])
            else:
                answers_start.append(None)
                answers_text.append(None)
            contexts.append(sub_para['context'])
            title.append(title_)
