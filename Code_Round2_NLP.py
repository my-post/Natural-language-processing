import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import RegexpParser
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import codecs
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import os
import re
from nltk.sem.relextract import extract_rels, rtuple
import seaborn as sn
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import ieer

def plot(X,Y,xlabel,ylabel,title):
    plt.bar(X, Y, tick_label = X, width = 0.4, color = ['orange', 'black'])
    plt.xticks(rotation=270)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

#To plot 10 most occured entries in distribution
def plotTop10Words(tokens,title):
    tags = []
    for i in tokens.keys():
        tags.append((tokens[i],i))
        
    tags.sort(reverse = True)
    X = []
    Y = []
    for i in tags:
        X.append(i[1])
        if len(X) ==10:
            break
        
    for i in tags:
        Y.append(i[0])
        if len(Y) == 10:
            break
        
    label_X = 'categories'
    label_Y = 'frequency'
    plot(X,Y,label_X,label_Y,title)

#Preprocessing of Books
def preprocessBook1():
    doc = codecs.open('Book1_Memoirs_of_Extraordinary_Popular_Delusions_and_the_Madness_of_Crowds.txt','r','utf-8')
    start = 'The Project Gutenberg EBook of Memoirs of Extraordinary Popular Delusions\r\nand the Madness of Crowds, by Charles Mackay\r\n\r\nThis eBook is for the use of anyone anywhere at no cost and with\r\nalmost no restrictions whatsoever.\n'
    end = '\n"My relative already spoken of rejoined'
    sentences = getSentences(doc,start,end,False)
    #print(sentences)
    return sentences

def preprocessBook2():
    doc = codecs.open('Book2_Pride_and_Prejudice.txt','r','utf-8')
    start = '\r\nThe Project Gutenberg EBook of Pride and Prejudice, by Jane Austen\r\n\r\nThis eBook is for the use of anyone anywhere at no cost and with\r\nalmost no restrictions whatsoever.\n'
    end = '\nBut,\r\n      perhaps, Mr. Bingley did not take the house so much for the\r\n      convenience of the neighbourhood as for his own, and we must\r\n      expect him to keep it or quit it on the same principle.”\r\n\r\n      “I should not be surprised,” said Darcy, “if he were to give it\r\n      up as soon as any eligible purchase offers.”\r\n\r\n      Elizabeth made'
    sentences = getSentences(doc,start,end,False)
    #print(sentences)
    return sentences

#returns pure form of sentences 
def getSentences(doc,start,end,test):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    content = doc.read()
    text = '\n@@\n'.join(tokenizer.tokenize(content))
    sentences = text.split('@@')
    #print(sentences)
    if not test:
        while True:
            if sentences[0] == start:
                sentences.pop(0)
                break
            sentences.pop(0)

        while True:
            if sentences[-1] == end:
                sentences.pop()
                break
            sentences.pop()

    ProcessedSentences = []
    for sentence in sentences:
        ProcessedSentences.append(sentence.replace('\n', ' ').replace('\r', '').replace('“', '').replace('”','').replace('—',''))

    return ProcessedSentences

#finds categories of verbs and nouns and there frewuency
def findCategories(tokens,tags,nouns,verbs):
    for word in tokens:
        if not lesk(tokens,word):
            continue
        if lesk(tokens, word).pos() == 'n':
            category = lesk(tokens, word).lexname()
            if category not in nouns.keys():
                nouns[category] = 1
            else:
                nouns[category] += 1
        elif lesk(tokens, word).pos() == 'v':
            category = lesk(tokens, word).lexname()
            if category not in verbs.keys():
                verbs[category] = 1
            else:
                verbs[category] += 1

#Plots the frequency distribution of all categories of nouns and verbs
def findVerbsAndNouns(sentences):
    nouns = {}
    verbs = {}

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        findCategories(tokens,tags,nouns,verbs)

    X = []
    Y = []
    for noun in nouns.keys():
        X.append(noun.split('.')[1][:4])
        Y.append(nouns[noun])

    #print(tags)
    xlabel = 'noun categories'
    ylabel = 'frequency'
    title = 'Relationship between noun categories and their frequency'
    title1 = 'Top 10 noun categories and their frequency'
    
    plot(X,Y,xlabel,ylabel,title)
    plotTop10Words(nouns, title1)

    X = []
    Y = []
    for verb in verbs.keys():
        X.append(verb.split('.')[1][:4])
        Y.append(verbs[verb])

    
    xlabel = 'verb categories'
    ylabel = 'frequency'
    title = 'Relationship between verb categories and their frequency'
    title1 = 'Top 10 verb categories and their frequency'

    plot(X,Y,xlabel,ylabel,title)
    plotTop10Words(verbs, title1)

#Forms entities for named entity recognition from the use of processed sentences
def namedEntityRecognition(sentences):
    entities = {}
    nlp = spacy.load('en')
    for sentence in sentences:
        doc = nlp(sentence)
        for X in doc.ents:
            if X.label_ not in entities.keys():
                entities[X.label_] = []
            entities[X.label_].append(X.text.lower())
    #print(entities)
    return entities
       
#Forms Named entity Realtions
def relationBetweenEntities(sentences):
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.tag.pos_tag(sentence) for sentence in tokenized_sentences]
    OF = re.compile(r'.*\bof\b.*')
    IN = re.compile(r'.*\bin\b(?!\b.+ing)')
    print('PERSON-ORGANISATION Relationships:')
    for i, sent in enumerate(tagged_sentences):
        sent = nltk.chunk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
        rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=IN, window=10) 
        for rel in rels:
            print(rtuple(rel))

    print('PERSON-GPE Relationships:')
    for i, sent in enumerate(tagged_sentences):
        sent = nltk.chunk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=OF, window=10) 
        for rel in rels:
            print(rtuple(rel))
            
#Plots the frequency distribution for different entities present in the text  
def NER(sentences):
    entities = namedEntityRecognition(sentences)
    X = []
    Y = []
    Xfig = []
    Yfig = []
    entityFig = {'PERSON', 'ORG', 'FAC', 'LOC', 'GPE'}
    for i in entities.keys():
        if i in entityFig:
            Xfig.append(i[:4])
            Yfig.append(len(entities[i]))
        X.append(i[:4])
        Y.append(len(entities[i]))

    xlabel = 'entities'
    ylabel = 'frequency'
    title = 'Relationship between entities and their frequency'
    
    plot(X,Y,xlabel,ylabel,title)
    plot(Xfig, Yfig, xlabel, ylabel, title)     #For the entities that are mentioned in Figure 22.1 

#Gives entities for the testing document (to be compared with the manual labelling)
def testingBook1():
    doc = codecs.open('testing_Book1.txt','r','utf-8')
    start = 'His bribe was refused, and he suffered the penalty of death.\n'
    end = '\nThe minister took great credit to himself for his\r\nshare in this transaction, and the scheme was always called by his\r\nflatterers "the Earl of Oxford\'s masterpiece."'
    sentences = getSentences(doc,start,end,False)
    entities = namedEntityRecognition(sentences)
    print(entities)

def testingBook2():
    doc = codecs.open('testing_Book2.txt','r','utf-8')
    start = 'The evening altogether passed off pleasantly to the whole family.\n'
    end = '\nShe had even condescended to\r\n      advise him to marry as soon as he could, provided he chose with\r\n      discretion; and had once paid him a visit in his humble\r\n      parsonage, where she had perfectly approved all the alterations\r\n      he had been making, and had even vouchsafed to suggest some\r\n      herself—some shelves in the closet up stairs.”'
    sentences = getSentences(doc,start,end,False)
    entities = namedEntityRecognition(sentences)
    print(entities)

#-------------------------------------------Driver Function------------------------------------------------

#Book-1 (Memoirs of Extraordinary Popular Delusions and the Madness of Crowds)
sent = preprocessBook1()                #To get the pure form of text(processed text)
findVerbsAndNouns(sent)                 #To find the categories of nouns and verbs and there frequency distribution
NER(sent)                               #For the named entity recognition and the frequency distribution accross all the entities as well as for the entitis particularly mentioned in Figure 22.5
relationBetweenEntities(sent)           #For generating the named entity relations in the books 
testingBook1()                          #For the named entity recognition of testing document for campairing with the manual labelling

#Book-2 (Pride and Prejudice)
sent = preprocessBook2()                #To get the pure form of text(processed text)
findVerbsAndNouns(sent)                 #To find the categories of nouns and verbs and there frequency distribution
NER(sent)                               #For the named entity recognition and the frequency distribution accross all the entities as well as for the entitis particularly mentioned in Figure 22.5
relationBetweenEntities(sent)           #For generating the named entity relations in the books 
testingBook1()                          #For the named entity recognition of testing document for campairing with the manual labelling

