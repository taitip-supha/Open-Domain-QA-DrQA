txt = "Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an entrepreneur and business magnate. He is the founder, CEO, and Chief Engineer at SpaceX; early stage investor,[note 1] CEO, and Product Architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI. A centibillionaire, Musk is one of the richest people in the world. ....... for his other views on such matters as artificial intelligence and public transport."

# importing the libraries
import numpy as np
import nltk
import re , math
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora

nltk.download('punkt')  #punkt is nltk tokenizer
tokens = nltk.sent_tokenize(txt) # txt contains the text/contents of your document.

def clean_sentence(sentence, stopwords=False):
 sentence = sentence.lower().strip()
 sentence = re.sub(r'[^a-z0-9s]', '', sentence)
 if stopwords:
   sentence = remove_stopwords(sentence)
 return sentence

def get_cleaned_sentences(tokens, stopwords=False):
 cleaned_sentences = []
 for line in tokens:
   cleaned = clean_sentence(line, stopwords)
   cleaned_sentences.append(cleaned)
 return cleaned_sentences

cleaned_sentences = get_cleaned_sentences(clean_sentence(tokens,True) ,True)
sentences = cleaned_sentences
sentence_words = [[word for word in document.split()]
                 for document in sentences]
dictionary = corpora.Dictionary(sentence_words)
# for key, value in dictionary.items():
#     print(key, ' : ', value)
corpus = [dictionary.doc2bow(text) for text in sentence_words]
for sent, embedding in zip(sentences, corpus):
    print(sent)
    print(embedding)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(cleaned_sentences)
tfidf_vectors=tfidfvectoriser.transform(cleaned_sentences)

def Euclidean(self, question_vector: np.ndarray, sentence_vector: np.ndarray):
    vec1 = question_vector.copy()
    vec2 = sentence_vector.copy()
    if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
    vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
    return np.linalg.norm(vec1-vec2)

def Cosine(self, question_vector: np.ndarray, sentence_vector: np.ndarray):
    dot_product = np.dot(question_vector, sentence_vector.T)
    denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
    return dot_product/denominator