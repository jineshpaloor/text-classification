import logging
import os
import wordcloud
import nltk
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TEXTS_DIR = "data/reuter_sample/texts"
MODELS_DIR = "data/models"
MAX_K = 10
NUM_TOPICS = 5
#stoplist = set(nltk.corpus.stopwords.words("english"))

f = open('data/stop-words/stop-words-english4.txt', 'r')
stoplist = [w.strip() for w in f.readlines() if w]
f.close()

def iter_wiki_docs(topdir):
    """
    iterate through the wikipedia xml file.
    understand the end of the article tag and consider it as one xml file.
    iterate through each yield one article tokens at a time.
    """
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)

class WikiCorpus(object):
    
    def __init__(self, topdir):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, stoplist))
    
    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

def iter_docs(topdir, stoplist):
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)

class MyCorpus(object):
    
    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, stoplist))
    
    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

def main():
    corpus = MyCorpus(TEXTS_DIR, stoplist)
    corpus.dictionary.save(os.path.join(MODELS_DIR, "mtsamples.dict"))
    gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "mtsamples.mm"), corpus)

    #dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "mtsamples.dict"))
    #corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))
    #gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=100).save(MODELS_DIR+'/lda.model')

def lsi_model():
    dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))
    
    tfidf = gensim.models.TfidfModel(corpus, normalize=True)
    corpus_tfidf = tfidf[corpus]
    
    # project to 2 dimensions for visualization
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    
    # write out coordinates to file
    fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
    for vector in lsi[corpus]:
        if len(vector) != 2:
            continue
        fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
    fcoords.close()

def num_topics():
    X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
    ks = range(1, MAX_K + 1)
    
    inertias = np.zeros(MAX_K)
    diff = np.zeros(MAX_K)
    diff2 = np.zeros(MAX_K)
    diff3 = np.zeros(MAX_K)
    for k in ks:
        kmeans = KMeans(k).fit(X)
        inertias[k - 1] = kmeans.inertia_
        # first difference
        if k > 1:
            diff[k - 1] = inertias[k - 1] - inertias[k - 2]
        # second difference
        if k > 2:
            diff2[k - 1] = diff[k - 1] - diff[k - 2]
        # third difference
        if k > 3:
            diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
    
    elbow = np.argmin(diff3[3:]) + 3
    
    plt.plot(ks, inertias, "b*-")
    plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
             markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
    plt.ylabel("Inertia")
    plt.xlabel("K")
    plt.show()

def topic_scatter():
    X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
    kmeans = KMeans(NUM_TOPICS).fit(X)
    y = kmeans.labels_
    
    colors = ["b", "g", "r", "m", "c"]
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
    plt.show()

def process_lda_print_topics(input_list):
    return [item.split('*') for text in input_list for item in text.split('+')]

def lda_model():
    dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))
    
    # Project to LDA space
    lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    output = lda.print_topics(NUM_TOPICS)
    print '******** output **********'
    print process_lda_print_topics(output)

def word_cloud():
    final_topics = open(os.path.join(MODELS_DIR, "final_topics.txt"), 'rb')
    curr_topic = 0
    for line in final_topics:
        line = line.strip()[line.rindex(":") + 2:]
        scores = [float(x.split("*")[0]) for x in line.split(" + ")]
        words = [x.split("*")[1] for x in line.split(" + ")]
        freqs = []
        for word, score in zip(words, scores):
            freqs.append((word, score))
        #elements = wordcloud.fit_words(freqs, width=120, height=120)
        wordcloud.draw(freqs, "gs_topic_%d.png" % (curr_topic), width=120, height=120)
        curr_topic += 1
    final_topics.close()



if __name__ == '__main__':
    #main()
    #lsi_model()
    #num_topics()
    lda_model()
    #word_cloud()
