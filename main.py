import json
import gensim
import numpy as np
import re
import string
from tqdm import tqdm

def readjson(filename): #Reads JSON data
    reader = open(filename)
    data = json.load(reader)
    reader.close()

    return data

def textclean(text): #Cleans text
    #Converts to lowercase, removes punctuation, unicode and newlines
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'https*\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'\'\w+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text

def corpusort(text): #Sorts input into documents and titles
    doc = []
    title = []
    for i in tqdm(range(len(text))):
        header = text[i][0]
        
        if header[0:3] == 'doc':
            doc.append(text[i])
        else:
            title.append(text[i])
    return doc,title

def cleandocs(docs): #Tokenises and preps the document data
    clean_docs = []
    for i in tqdm(range(len(docs))):
        tokens = gensim.utils.simple_preprocess(textclean(docs[i][1]))
        clean_docs.append(gensim.models.doc2vec.TaggedDocument(tokens,docs[i][0]))

    return clean_docs

def cleantitles(titles): #Tokenises and preps the title data
    clean_titles = []
    for i in tqdm(range(len(titles))):
        clean_titles.append([gensim.utils.simple_preprocess(textclean(titles[i][1])),titles[i][0]])
    
    return clean_titles

corpus = readjson('data/corpus.json')
docs, titles = corpusort(corpus)

docs = cleandocs(docs)
titles = cleantitles(titles)

model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, epochs = 40)
model.build_vocab(docs)

print (clean_docs)