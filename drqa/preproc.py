import re
import torch
import pickle
import collections
import spacy_thai
import unicodedata
from train import BatchGen

class argumentParser(dict):
  def __init__(self, **kwargs):
    for k in kwargs.keys():
         self.__setattr__(k, kwargs[k])

def save_to_pickle(save_obj, path_file):
    with open(path_file, 'wb') as file:
        pickle.dump(save_obj, file)
    print(f"save to {path_file} success")

def load_pickle(path_file):
    with open(path_file, 'rb') as file:
        load_obj = pickle.load(file)
        print(f"load object from {path_file} success,that is {type(load_obj)}")
        return load_obj

def annotate(row, wv_cased):
    id_, context, question = row[:3]
    q_doc = nlp(question)
    c_doc = nlp(context)
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span) + row[3:]
  
def index_answer(row):
    token_span = row[-4]
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    try:
        return row[:-3] + (starts.index(answer_start), ends.index(answer_end))
    except ValueError:
        return row[:-3] + (None, None)

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def init():
  global nlp
  """initialize spacy in each process"""
  nlp = spacy_thai.load()

def build_vocab(questions, contexts, wv_vocab, sort_all=False):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter

def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]

def load_data(opt):
  with open(opt['meta_file'], 'rb') as file:
    meta = pickle.load(file, encoding='utf8')
  embedding = torch.Tensor(meta['embedding'])
  opt['pretrained_words'] = True
  opt['vocab_size'] = embedding.size(0)
  opt['embedding_dim'] = embedding.size(1)
  opt['pos_size'] = len(meta['vocab_tag'])
  opt['ner_size'] = len(meta['vocab_ent'])
  BatchGen.pos_size = opt['pos_size']
  BatchGen.ner_size = opt['ner_size']
  with open(opt['data_file'], 'rb') as f:
    data = pickle.load(f, encoding='utf8')
  train = data['train']
  dev = data['dev']
  data['dev'].sort(key=lambda x: len(x[1]))
  dev_x = [x[:-1] for x in data['dev']]
  dev_y = [x[-1] for x in data['dev']]
  return train, dev, dev_x, dev_y, embedding, opt
