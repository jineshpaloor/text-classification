import time
import os
import wikipedia
import gensim
import logging

module_logger = logging.getLogger('wiki_module_logger')
# set file handler
fh = logging.FileHandler('data/wiki_module.log')
fh.setLevel(logging.DEBUG)
FORMAT = '%(asctime)-15s :: %(message)s'
formatter = logging.Formatter(FORMAT)
fh.setFormatter(formatter)
module_logger.addHandler(fh)

DICT_PATH = "data/wiki.dict"
MODEL_PATH = "data/wiki_normal.lda"
OUTPUT_PATH = "data/output_normal.txt"
WIKI_PATH = "data/wiki_download_dir"

f = open('data/stop-words/stop-words-english4.txt', 'r')
STOPLIST = [w.strip() for w in f.readlines() if w]
f.close()


class TagWiki(object):
    def __init__(self, distributed=False):
        log_file = 'data/wiki_{0}.log'.format('distributed' if distributed else 'normal')
        self.logger = logging.getLogger('wiki_log')
        ch = logging.FileHandler(log_file)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.electrical_links = []
        self.wiki_path = WIKI_PATH
        self.lda = None
        self.distributed = distributed

        # initialize dictionary
        if os.path.exists(DICT_PATH):
            self.dictionary = gensim.corpora.Dictionary.load(DICT_PATH)
        else:
            self.dictionary = gensim.corpora.Dictionary()

    def _init_lda(self):
        """ initialize lda model. This should be called only after the dictionary is prepared.
        Otherwise dictionary saved to a file should be ready beforehand.
        """
        if os.path.exists(MODEL_PATH):
            self.lda = gensim.models.ldamodel.LdaModel.load(MODEL_PATH)
        else:
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=None, id2word=self.dictionary, num_topics=30,
                update_every=1, chunksize=1, passes=10, distributed=self.distributed)

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
        for fn in os.listdir(self.wiki_path):
            content = self.get_processed_content(fn)
            self.dictionary.add_documents([content])
        self.dictionary.save(DICT_PATH)
        return True

    def get_sorted_topics(self, bow):
        """
        take bow as input and return back the relevent topics for it
        """
        return sorted(self.lda[bow], key=lambda x: x[1], reverse=True)

    # Pass 2: Process topics
    def process_topics_from_docs(self):
        """
        Read documents from wikipedia articles in data folder and then
          - update lda model
          - predict the relevent topics for the document
        """
        self._init_lda()
        f = open(OUTPUT_PATH, "w")
        for fn in os.listdir(self.wiki_path):
            try:
                self.logger.info("processing {0}".format(fn))
                content = self.get_processed_content(fn)
                content_bow = self.dictionary.doc2bow(content)
                self.lda.update([content_bow])
                topics = self.get_sorted_topics(content_bow)
                f.write("{0}::    {1}\n".format(fn, topics))
            except UnicodeError:
                self.logger.info("PROCESSING FAILED!")
                continue
        f.close()
        self.lda.save(MODEL_PATH)
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
        self.dictionary.save(DICT_PATH)
        return True

    # Pass 2: Process topics
    def process_topics(self):
        """
        Read documents using wikipedia library api and then
          - update lda model
          - predict the relevent topics for the document
        """
        self._init_lda()
        f = open(OUTPUT_PATH, "w")
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
        self.lda.save(MODEL_PATH)
        return True

def main():
    start_time = time.time()
    module_logger.info("START TIME :{0}".format(start_time))

    wiki = TagWiki(distributed=False)
    module_logger.info("No. of keys at start :{0}".format(wiki.dictionary.keys().__len__()))

    wiki.prepare_dictionary_from_docs()
    dict_prepare_time = time.time()
    module_logger.info("TIME AFTER DICTIONARY PREPARATION :{0}".format(dict_prepare_time))

    wiki.process_topics_from_docs()
    first_pass = time.time()
    module_logger.info("TIME AFTER FIRST PASS :{0}".format(first_pass))

    wiki.process_topics_from_docs()
    second_pass = time.time()
    module_logger.info("TIME AFTER SECOND PASS :{0}".format(second_pass))

    total_time = (start_time - second_pass) / 60
    module_logger.info("TOTAL TIME ELAPSED :{0}".format(total_time))
    module_logger.info("TOPICS:{0}".format(wiki.lda.print_topics()))


if __name__ == '__main__':
    main()
