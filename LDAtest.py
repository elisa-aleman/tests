#-*- coding: utf-8 -*-

import os.path
import sys
PythonPath = os.path.join(os.path.expanduser('~'), 'Master','python','libraries')
sys.path.append(os.path.abspath(PythonPath))
from ProjectMethods import *
from RemoveNoise import *
import gensim

def getCorpus(c, limit=None):
    sql = "SELECT Sentence FROM Sentences_NEW"
    if limit:
        sql = "SELECT DISTINCT Sentence FROM Sentences_NEW ORDER BY Random() LIMIT {}".format(limit)
    c.execute(sql)
    raw = c.fetchall()
    corpus = [i[0] for i in raw]
    return corpus

def Tokenize(corpus):
    tokenized = [i.split() for i in corpus]
    counts = dict()
    for sentence in tokenized:
        for word in sentence:
            if counts.get(word):
                counts[word] += 1
            else:
                counts[word] = 1
    tokenized = [[word for word in sentence if counts.get(word)>1] for sentence in tokenized]
    return tokenized

def getDictionary(tokenized):
    # dictionary = list(set(flatten(corpus)))
    dictionary = gensim.corpora.Dictionary(tokenized)
    dictionary.save(MakeLogFile("LDAtestdictionary.dict"))
    return dictionary

def Vectorize(tokenized,dictionary,storage=MakeLogFile("LDAtestVector.mm")):
    #Returns sparse vectors
    vectorizedcorpus = [dictionary.doc2bow(sentence) for sentence in tokenized]
    #Stores in file
    gensim.corpora.MmCorpus.serialize(storage,vectorizedcorpus)
    return vectorizedcorpus

def LDA(vectorizedcorpus, num_topics, dictionary):
    lda = gensim.models.ldamodel.LdaModel(corpus=vectorizedcorpus, num_topics=num_topics, id2word=dictionary)
    return lda

def test():
    print("Connecting to SQL Database")
    conn, c = Connect("ctrip_db.sqlite")
    print("Getting corpus from SQL")
    corpus = getCorpus(c, 1000)
    print("Tokenizing Corpus")
    tokenized = Tokenize(corpus)
    print("Creating statistical dictionary")
    dictionary = getDictionary(tokenized)
    print("Vectorizing corpus with dictionary")
    vectorizedcorpus = Vectorize(tokenized, dictionary, storage=MakeLogFile("LDAtestVector_sample1000.mm"))
    ###
    print("Creating LDA Model")
    lda = LDA(vectorizedcorpus, 100, dictionary)
    print("Topics")
    topics = lda.print_topics(100)
    topiclog = MakeLogFile("LDAtest_100topics_sample1000.txt")
    for i in topics:
        ustr = u"topic {}: {}".format(i[0],i[1])
        printSTDlog(ustr, topiclog)


def main():
    print("Connecting to SQL Database")
    conn, c = Connect("ctrip_db.sqlite")
    print("Getting corpus from SQL")
    corpus = getCorpus(c)
    print("Tokenizing Corpus")
    tokenized = Tokenize(corpus)
    print("Creating statistical dictionary")
    dictionary = getDictionary(tokenized)
    print("Vectorizing corpus with dictionary")
    vectorizedcorpus = Vectorize(tokenized, dictionary, storage=MakeLogFile("LDAtestVector_fullCorpus.mm"))
    ###
    print("Creating LDA Model")
    lda = LDA(vectorizedcorpus, 100, dictionary)
    print("Topics")
    topics = lda.print_topics(100)
    topiclog = MakeLogFile("LDAtest_100topics_fullCorpus.txt")
    for i in topics:
        ustr = u"topic {}: {}".format(i[0],i[1])
        printSTDlog(ustr, topiclog)

if __name__ == "__main__":
    main()
    # test()
