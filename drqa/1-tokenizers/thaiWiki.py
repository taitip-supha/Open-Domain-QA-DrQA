import pandas as pd
import numpy as np
import json, re, unicodedata, string, typing, time
import pickle
from bs4 import BeautifulSoup


def load_json(path):
    '''
    Loads the JSON file of the Squad dataset.
    Returns the json object of the dataset.
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("Length of data: ", len(data['data']))
    print("Data Keys: ", data['data'][0].keys())
    print("Title: ", data['data'][0]['title'])
    
    return data

#Open Json File
with open('./data/ThaiQACorpus-DevelopmentDataset.json', 'r', encoding='utf-8') as f:
  data = json.load(f)
print("Length of data: ", len(data['data']))
print("Data Keys: ", data['data'][0].keys())


with open('./data/documents-nsc/616.txt', 'r', encoding='utf-8') as f:
  cxt_text = BeautifulSoup(f.read(), 'html.parser')

type(cxt_text)