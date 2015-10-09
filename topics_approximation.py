import logging
import os
import wordcloud
import nltk
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
FORMAT = '%(asctime)-15s :: %(levelname)s :: %(name)s :: %(message)s'
formatter = logging.Formatter(FORMAT)

def iter_docs(topdir, stoplist):
    """
    iterate through the files in a folder and tokenize each document
    """
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)


class MyCorpus(object):
    def __init__(self, topdir):
        self.topdir = topdir
        self.MAX_K = 100
        self.NUM_TOPICS = 10
        self.MODELS_DIR  = "data/models"
        self.CORPUS_PATH = self.MODELS_DIR + "/wiki.mm"
        self.DICT_PATH = self.MODELS_DIR + "/wiki.dict"
        self.COORDS_PATH = self.MODELS_DIR + "/coords.csv"

        self.stoplist = set(nltk.corpus.stopwords.words("english"))
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, self.stoplist))
        self.corpus = []
        # adding logger
        log_file = "data/topics_log.log"
        self.logger = logging.getLogger('wiki_log')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.FileHandler(log_file)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("wiki topics approximation initialized")

    def bow_docs(self):
        self.logger.info("bow docs ..")
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

    def save_corpus(self):
        self.logger.info("saving corpus")
        gensim.corpora.MmCorpus.serialize(self.CORPUS_PATH, self.bow_docs())

    def set_corpus(self):
        self.logger.info("setting corpus instance variable")
        self.corpus = gensim.corpora.MmCorpus(self.CORPUS_PATH)
    
    def save_dict(self):
        self.logger.info("saving dict..")
        if os.path.exists(self.DICT_PATH):
            return True
        self.dictionary.save(self.DICT_PATH)
        self.logger.info("saving dict completed..")

    def _get_lsi(self):
        self.logger.info("get lsi..")
        tfidf = gensim.models.TfidfModel(self.corpus, normalize=True)
        corpus_tfidf = tfidf[self.corpus]
        return gensim.models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=2)

    def save_vectors(self):
        if os.path.exists(self.COORDS_PATH):
            return True
        self.logger.info("writing vector")
        lsi = self._get_lsi()
        fcoords = open(self.COORDS_PATH, 'wb')
        vectors = lsi[self.corpus]
        for vector in vectors:
            if len(vector) != 2:
                continue
            fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
        fcoords.close()

    def plot_kmean(self):
        self.logger.info("calculating k-means..")
        self.logger.info("")
        X = np.loadtxt(self.COORDS_PATH, delimiter="\t")
        ks = range(1, self.MAX_K + 1)
        
        inertias = np.zeros(self.MAX_K)
        diff = np.zeros(self.MAX_K)
        diff2 = np.zeros(self.MAX_K)
        diff3 = np.zeros(self.MAX_K)
        for k in ks:
            self.logger.info(k)
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
        
        f = open("data/map.txt", "w")
        print ks
        print list(inertias)
        print elbow
        f.write(str(ks))
        f.write("\n")
        f.write(str(list(inertias)))
        f.write("\n")
        f.write(str(elbow))
        f.close()

        self.logger.info("k means create image")
        plt.plot(ks, inertias, "b*-")
        plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
        plt.ylabel("Inertia")
        plt.xlabel("K")
        plt.show()

    def topic_scatter(self):
        self.logger.info("topic scatter..")
        X = np.loadtxt(self.COORDS_PATH, delimiter="\t")
        kmeans = KMeans(self.NUM_TOPICS).fit(X)
        y = kmeans.labels_
        
        colors = ["b", "g", "r", "m", "c"]
        for i in range(X.shape[0]):
            plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
        plt.show()


if __name__ == '__main__':
    topdir = "/Users/jineshn/Code/python/gensim_scripts/data/wiki_electrical"
    wiki_corpus = MyCorpus(topdir)
    wiki_corpus.save_dict()
    wiki_corpus.save_corpus()
    wiki_corpus.set_corpus()
    wiki_corpus.save_vectors()
    wiki_corpus.plot_kmean()
    #wiki_corpus.topic_scatter()
