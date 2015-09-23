import time
import os
import sys
import logging
import itertools
import gc

import numpy as np
import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

# set module lever logger
FORMAT = '%(asctime)-15s :: %(levelname)s :: %(name)s :: %(message)s'
formatter = logging.Formatter(FORMAT)

class ProcessWiki(object):

    def __init__(self, dump_file, distributed):
        self.dump_file = dump_file
        self.dictionary = gensim.corpora.Dictionary([])
        self.clip_docs = 5

        if distributed:
            vt = 'distributed'
        else:
            vt = 'normal'

        self.OUTPUT_PATH = "/mnt/data/logs/output_{0}.txt".format(vt)
        self.DICT_PATH = "/mnt/data/logs/wiki_dump_{0}.dict".format(vt)
        self.MODEL_PATH = "/mnt/data/logs/wiki_dump_{0}.lda".format(vt)
        log_file = '/mnt/data/logs/wiki_dump_{0}.log'.format(vt)

        self.logger = logging.getLogger('wiki_log')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.FileHandler(log_file)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("Tag wiki initialized")

        self.lda = None
        self.distributed = distributed

        # initialize dictionary
        if os.path.exists(self.DICT_PATH):
            self.dictionary = gensim.corpora.Dictionary.load(self.DICT_PATH)
        else:
            self.dictionary = gensim.corpora.Dictionary()

    def iter_wiki(self):
        """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
        for title, text, pageid in _extract_pages(smart_open(self.dump_file)):
            text = filter_wiki(text)
            tokens = [token for token in simple_preprocess(text) if token not in STOPWORDS]
            if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
                continue  # ignore short articles and various meta-articles
            yield title, tokens

    def _init_lda(self):
        """ initialize lda model. This should be called only after the dictionary is prepared.
        Otherwise dictionary saved to a file should be ready beforehand.
        """
        if False: #os.path.exists(self.MODEL_PATH):
            self.lda = gensim.models.ldamodel.LdaModel.load(self.MODEL_PATH)
        else:
            # chunksize determines the number of documents to be processed in a worker.
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=None, id2word=self.dictionary, num_topics=100,
                update_every=10, chunksize=10, passes=10, distributed=self.distributed)

    # Pass 1: Prepare Dictionary
    def prepare_dictionary_from_docs(self):
        """
        iterate through the wikipedia docs dir. and update dictionary
        """
        self.logger.info("START PREPARING DICT")
        for title, tokens in self.iter_wiki():
            try:
                self.logger.info("{0} dict update {1}".format(counter, title))
                self.dictionary.add_documents([tokens])
                self.dictionary.save(self.DICT_PATH)
            except UnicodeError:
                continue
        return True

    # Pass 2: Process topics
    def update_lda_model(self):
        """
        Read documents from wikipedia articles in data folder and then
          - update lda model
          - predict the relevent topics for the document
        """
        self.logger.info("START UPDATING LDA")
        self._init_lda()
        counter = 0
        bow_list = []
        for title, tokens in itertools.islice(self.iter_wiki(), self.clip_docs):
            try:
                self.logger.info("updating lda: {0}".format(title))
                bow = self.dictionary.doc2bow(tokens)
                bow_list.append(bow)
                if counter == 5:
                    self.lda.update(bow_list)
                    counter = 0
                    bow_list = []
                else:
                    counter += 1
            except UnicodeError:
                self.logger.info("PROCESSING FAILED!")
                continue
        self.lda.save(self.MODEL_PATH)
        return True

    # Pass 3: Print topic for each document
    def print_document_topics(self):
        self.logger.info("START PRINTING DOCUMENTS")
        for title, tokens in self.iter_wiki():
            try:
                # get the topics for files and write it to log file
                bow = self.dictionary.doc2bow(tokens)
                topics = sorted(self.lda[bow], key=lambda x: x[1], reverse=True)
                topic = self.lda.print_topic(topics[0][0])
                self.logger.info("{0} :: {1}\n".format(title, topic))
            except UnicodeError:
                pass
        return True


def main(wiki_path, run_type):
    if run_type.lower() not in ['true', 'false']:
        print 'Invalid input'
        sys.exit(0)
    if run_type.lower() == 'true':
        distributed = True
        fn = '/mnt/data/logs/wiki_dump_module_{0}.log'.format('distributed')
    else:
        distributed = False
        fn = '/mnt/data/logs/wiki_dump_module_{0}.log'.format('normal')

    logging.basicConfig(filename=fn, level=logging.DEBUG, format=FORMAT)
    module_logger = logging.getLogger('wiki_module_logger')
    module_logger.setLevel(logging.DEBUG)
    # set file handler
    fh = logging.FileHandler(fn)
    fh.setLevel(logging.DEBUG)

    fh.setFormatter(formatter)
    module_logger.addHandler(fh)

    start_time = time.time()

    module_logger.info("START TIME :{0}".format(start_time))
    wiki = ProcessWiki(wiki_path, distributed)

    # PASS 1
    wiki.prepare_dictionary_from_docs()
    dict_prepare_time = time.time()
    module_logger.info("TIME AFTER DICTIONARY PREPARATION :{0}".format(dict_prepare_time))

    # PASS 2
    wiki.update_lda_model()
    first_pass = time.time()
    module_logger.info("TIME AFTER FIRST PASS :{0}".format(first_pass))

    # PASS 3
    wiki.print_document_topics()
    second_pass = time.time()
    module_logger.info("TIME AFTER DOC PRINT  :{0}".format(second_pass))

    total_time = (second_pass - start_time) / 60
    module_logger.info("TOTAL TIME ELAPSED :{0}".format(total_time))

def create_wiki_dict(wiki_path, run_type):
    from gensim.corpora.wikicorpus import WikiCorpus

    fn = 'logs/create_wiki_dict.log'
    logging.basicConfig(filename=fn, level=logging.DEBUG, format=FORMAT)
    module_logger = logging.getLogger('wiki_module_logger')
    module_logger.setLevel(logging.DEBUG)
    # set file handler
    fh = logging.FileHandler(fn)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    module_logger.addHandler(fh)

    module_logger.info("START")
    wiki_corpus = WikiCorpus(wiki_path)  # This will take many hours! Output is Wikipedia in bucket-of-words (BOW) sparse matrix.
    module_logger.info("Wiki corpus ready")
    wiki_corpus.dictionary.save("logs/wiki_dump_dict.dict")
    module_logger.info("Dictionary Created")

if __name__ == '__main__':
    wiki_path = sys.argv[1]
    run_type = sys.argv[2]
    #main(wiki_path, run_type)
    create_wiki_dict(wiki_path, run_type)
