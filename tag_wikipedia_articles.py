import time
import sys
import os
import wikipedia
import gensim
import logging

# set module lever logger
FORMAT = '%(asctime)-15s :: %(levelname)s :: %(name)s :: %(message)s'
formatter = logging.Formatter(FORMAT)

WIKI_PATH = "data/wiki_download_dir"
#WIKI_PATH = "data/bbc/business"
f = open('data/stop-words/stop-words-english4.txt', 'r')
STOPLIST = [w.strip() for w in f.readlines() if w]
f.close()


class TagWiki(object):
    def __init__(self, topics, update_every, chunk, passes, distributed=False):
        self.num_topics = topics
        self.update_every = update_every
        self.chunksize = chunk
        self.passes = passes

        if distributed:
            self.DICT_PATH = "data/wiki_distributed.dict"
            self.OUTPUT_PATH = "data/output_distributed_{0}_chunk_{1}_passes_{2}.txt".format(self.num_topics, self.chunksize, self.passes)
            self.MODEL_PATH = "data/wiki_distributed_topics_{0}_chunk_{1}_passes_{2}.lda".format(self.num_topics, self.chunksize, self.passes)
            log_file = 'data/wiki_distributed.log'.format(self.num_topics, self.chunksize, self.passes)
        else:
            self.DICT_PATH = "data/wiki_normal.dict"
            self.OUTPUT_PATH = "data/output_normal_{0}_chunk_{1}_passes_{2}.txt".format(self.num_topics, self.chunksize, self.passes)
            self.MODEL_PATH = "data/wiki_normal_topics_{0}_chunk_{1}_passes_{2}.lda".format(self.num_topics, self.chunksize, self.passes)
            log_file = 'data/wiki_normal_topics_{0}_chunk_{1}_passes_{2}.log'.format(self.num_topics, self.chunksize, self.passes)
        self.logger = logging.getLogger('wiki_log')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.FileHandler(log_file)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info("Tag wiki initialized")

        self.electrical_links = []
        self.wiki_path = WIKI_PATH
        self.lda = None
        self.distributed = distributed

        # initialize dictionary
        if os.path.exists(self.DICT_PATH):
            self.dictionary = gensim.corpora.Dictionary.load(self.DICT_PATH)
        else:
            self.dictionary = gensim.corpora.Dictionary()

    def _init_lda(self):
        """ initialize lda model. This should be called only after the dictionary is prepared.
        Otherwise dictionary saved to a file should be ready beforehand.
        """
        if False: #os.path.exists(self.MODEL_PATH):
            self.lda = gensim.models.ldamodel.LdaModel.load(self.MODEL_PATH)
        else:
            # chunksize determines the number of documents to be processed in a worker.
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=None, id2word=self.dictionary, num_topics=self.num_topics,
                update_every=self.update_every, chunksize=self.chunksize,
                passes=self.passes, distributed=self.distributed)

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

    def get_processed_content(self, fn):
        """
        Read a document from file, remove punctuation, stopwords etc and return list of words
        """
        fin = open(os.path.join(self.wiki_path, fn), 'rb')
        text = fin.read()
        fin.close()
        return (x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in STOPLIST)

    def prepare_dictionary_from_docs(self):
        """
        iterate through the wikipedia docs dir. and update dictionary
        """
        if os.path.exists(self.DICT_PATH):
            return True
        self.logger.info("START PREPARING DICT")
        for fn in os.listdir(self.wiki_path):
            self.logger.info("dict update {0}".format(fn))
            content = self.get_processed_content(fn)
            self.dictionary.add_documents([content])
        self.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=100000)
        self.dictionary.compactify()
        self.dictionary.save(self.DICT_PATH)
        return True

    def get_sorted_topics(self, bow):
        """
        take bow as input and return back the relevent topics for it
        """
        return sorted(self.lda[bow], key=lambda x: x[1], reverse=True)

    def get_bow_for_files(self, fns):
        bow_list = []
        for fn in fns:
            self.logger.info("processing: {0}".format(fn))
            content = self.get_processed_content(fn)
            content_bow = self.dictionary.doc2bow(content)
            bow_list.append(content_bow)
        return bow_list

    # Pass 2: Process topics
    def update_lda_model(self):
        """
        Read documents from wikipedia articles in data folder and then
          - update lda model
          - predict the relevent topics for the document
        """
        self.logger.info("START UPDATING LDA")
        self._init_lda()
        if os.path.exists(self.MODEL_PATH):
            return True
        file_names = os.listdir(self.wiki_path)
        for fns in self.chunks(file_names, 5):
            # update lda for files
            try:
                self.logger.info("updating lda: {0}".format(fns))
                content_bow_list = self.get_bow_for_files(fns)
                self.lda.update(content_bow_list)
            except UnicodeError:
                self.logger.info("PROCESSING FAILED!")
                continue
        self.lda.save(self.MODEL_PATH)
        return True

    def get_text_topic(self, topic):
        return self.lda.print_topic(topic[0])

    def print_document_topics(self):
        self.logger.info("START PRINTING DOCUMENTS")
        file_names = os.listdir(self.wiki_path)
        f = open(self.OUTPUT_PATH, "w")
        for fn in file_names:
            # get the topics for files and write it to log file
            content = self.get_processed_content(fn)
            content_bow = self.dictionary.doc2bow(content)
            topics = self.get_sorted_topics(content_bow)
            #topic = self.get_text_topic(topics[0])
            self.logger.info("{0} :: {1}\n".format(fn, topics))
            f.write("{0}: {1}\n".format(fn, topics))
        f.close()
        return True

    # Pass 1: Prepare a dictionary
    def prepare_dictionary(self):
        self.electrical_links = wikipedia.page("Index_of_electrical_engineering_articles").links
        for link in self.electrical_links:
            try:
                page = wikipedia.page(link)
                self.logger.info(link)
                title = gensim.parsing.preprocess_string(page.title)
                content = gensim.parsing.preprocess_string(page.content)
                self.dictionary.add_documents([title, content])
            except UnicodeError:
                self.logger.info("failed: {0}".format(link))
                continue
        self.dictionary.save(self.DICT_PATH)
        return True

    # Pass 2: Process topics
    def process_topics(self):
        """
        Read documents using wikipedia library api and then
          - update lda model
          - predict the relevent topics for the document
        """
        self._init_lda()
        f = open(self.OUTPUT_PATH, "w")
        for link in self.electrical_links:
            try:
                self.logger.info("processing: {0}".format(link))
                page = wikipedia.page(link)
                title = gensim.parsing.preprocess_string(page.title)
                content = gensim.parsing.preprocess_string(page.content)

                title_bow = self.dictionary.doc2bow(title)
                content_bow = self.dictionary.doc2bow(content)

                new_bag_of_words = title_bow + content_bow
                self.lda.update([content_bow])
                topics = self.get_sorted_topics(new_bag_of_words)
                f.write("{0}::    {1}\n".format(link, topics))
            except UnicodeError:
                self.logger.info("PROCESSING FAILED!")
                continue

        f.close()
        self.lda.save(self.MODEL_PATH)
        return True

def main(topics, update_every, chunk, passes):
    run_type = sys.argv[1]
    if run_type.lower() not in ['true', 'false']:
        print 'Invalid input'
        sys.exit(0)
    if run_type.lower() == 'true':
        distributed = True
        fn = 'data/wiki_module_{0}.log'.format('distributed')
    else:
        distributed = False
        fn = 'data/wiki_module_{0}.log'.format('normal')

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

    wiki = TagWiki(topics, update_every, chunk, passes, distributed=distributed)
    module_logger.info("No. of keys at start :{0}".format(wiki.dictionary.keys().__len__()))

    wiki.prepare_dictionary_from_docs()
    dict_prepare_time = time.time()
    module_logger.info("TIME AFTER DICTIONARY PREPARATION :{0}".format(dict_prepare_time))

    wiki.update_lda_model()
    first_pass = time.time()
    module_logger.info("TIME AFTER FIRST PASS :{0}".format(first_pass))

    wiki.print_document_topics()
    second_pass = time.time()
    module_logger.info("TIME AFTER DOC PRINT  :{0}".format(second_pass))

    total_time = (start_time - second_pass) / 60
    module_logger.info("TOTAL TIME ELAPSED :{0}".format(total_time))
    module_logger.info("TOPICS:{0}".format(wiki.lda.print_topics()))


if __name__ == '__main__':
    data = [
        (30, 10, 30, 10),
        (30, 10, 50, 10),
        (20, 10, 100, 10),
        (10, 10, 100, 10),
        (100, 10, 10, 10),
        (30, 10, 10, 100),
    ]
    for topics, update_every, chunk, passes in data:
        main(topics, update_every, chunk, passes)
