import sys
import os
from gensim import corpora, models, similarities, models

##import logging
##logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
##
##
doc_dict = {
        0: '/Users/jineshn/Documents/Sample/biology.txt',
        1: '/Users/jineshn/Documents/Sample/brain.txt',
        2: '/Users/jineshn/Documents/Sample/computer.txt',
        3: '/Users/jineshn/Documents/Sample/electonics.txt',
        4: '/Users/jineshn/Documents/Sample/energy.txt',
        5: '/Users/jineshn/Documents/Sample/info.txt',
        6: '/Users/jineshn/Documents/Sample/mind.txt',
        7: '/Users/jineshn/Documents/Sample/physics.txt',
        8: '/Users/jineshn/Documents/Sample/psycology.txt',
        9: '/Users/jineshn/Documents/Sample/science.txt'
}

class MyCorpus(object):
    def __init__(self):
        self.dictionary = None
        self.corpus = None
        self.sample_docs_path = '/Users/jineshn/Documents/Sample/'
        self.dict_path = '/tmp/sample.dict'
        self.corpus_path = '/tmp/sample.mm'
        self.stoplist = set('for a of the and to in * - ='.split())

    def __iter__(self):
        for dirpath, dirname, filenames in os.walk(self.sample_docs_path):
            # assume there's one document per line, tokens separated by whitespace
            all_files = [os.path.join(dirpath, fn) for fn in filenames if fn[0] != '.']
            for filename in all_files:
                with open(filename, 'r') as f:
                    words = f.readlines()
                    yield self.dictionary.doc2bow(words)

    def read_words(self):
        """Iterate/walk through sample files and return list of lines from each file at once"""
        for dirpath, dirname, filenames in os.walk(self.sample_docs_path):
            # assume there's one document per line, tokens separated by whitespace
            all_files = [os.path.join(dirpath, fn) for fn in filenames if fn[0] != '.']
            for ind, filename in enumerate(all_files):
                print ind, filename
                with open(filename, 'r') as f:
                    yield f.readlines()

    def remove_stop_words_from_dict(self):
        # remove stop words and words that appear only once
        stop_ids = [self.dictionary.token2id[stopword] for stopword in self.stoplist if stopword in self.dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems() if docfreq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        self.dictionary.compactify() # remove gaps in id sequence after words that were removed

    def set_dictionary(self):
        # collect statistics about all tokens
        words_list = [line.split() for lines_list in self.read_words() for line in lines_list if line ]
        self.dictionary = corpora.Dictionary(words_list)
        self.remove_stop_words_from_dict()

    def get_dictionary(self):
        return self.dictionary

    def get_vector(self, file_path):
        """Return the vector for input filepath"""
        with open(file_path) as f:
            sentence_list = [word.split() for word in f.readlines()]
            word_list = [item for sentence in sentence_list for item in sentence if item not in self.stoplist]
            return self.dictionary.doc2bow(word_list)

    def get_vector_for_sentence(self, sentence):
        """Return the vector for input sentence"""
        words = sentence.split('')
        word_list = [item for item in words if item not in self.stoplist]
        return self.dictionary.doc2bow(word_list)

    def get_corpus(self):
        return self.corpus

    def set_corpus(self):
        self.corpus = [
            self.get_vector(os.path.join(dirpath, fn)) 
            for dirpath, dirname, filenames in os.walk(self.sample_docs_path) 
            for fn in filenames if fn[0] != '.']

    def write_dict_to_file(self):
        self.dictionary.save(self.dict_path)

    def write_corpus_to_file(self):
        """ store to disk, for later use """
        corpora.MmCorpus.serialize(self.corpus_path, self.corpus)

def setup():
    cc = MyCorpus() # doesn't load the corpus into memory!
    cc.set_dictionary()
    cc.set_corpus()
    cc.write_dict_to_file()
    cc.write_corpus_to_file()

def read_semantical_match(file_path):
    # read dictionary and corpus
    dictionary = corpora.Dictionary.load('/tmp/sample.dict')
    corpus = corpora.MmCorpus('/tmp/sample.mm') # comes from the first tutorial, "From strings to vectors"
    
    v = cc.get_vector(file_path)
    tfidf = models.TfidfModel(corpus)

    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    print '*' * 100
    print lsi.print_topics(2)


def matching_docs(query_term):
    # read dictionary and corpus
    dictionary = corpora.Dictionary.load('/tmp/sample.dict')
    corpus = corpora.MmCorpus('/tmp/sample.mm') # comes from the first tutorial, "From strings to vectors"

    # define a 2 dimensional LSI space with the corpus
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    # get vector for input query_term
    vec_bow = dictionary.doc2bow(query_term.lower().split())
    # convert the query to LSI space
    vec_lsi = lsi[vec_bow] 
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[corpus]) 

    index.save('/tmp/sample.index')
    index = similarities.MatrixSimilarity.load('/tmp/sample.index')

    sims = index[vec_lsi] # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print sorted (document number, similarity score) 2-tuples
    for s in sims:
        print(s) , doc_dict.get(s[0])

def search_words():
    ls = ["mind", "computer", "information", "electron", "physics"]
    for x in ls:
        print '*' * 50
        print x
        print '*' * 50
        matching_docs(x)

setup()
search_words()
