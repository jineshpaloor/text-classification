import time
import os
import wikipedia
import gensim
import logging

logging.basicConfig(filename='wiki_normal.log', level=logging.INFO)

DICT_PATH = "data/wiki.dict"
MODEL_PATH = "data/wiki.lda"
OUTPUT_PATH = "data/output_normal.txt"
WIKI_PATH = "data/wiki_download_dir"

f = open('data/stop-words/stop-words-english4.txt', 'r')
STOPLIST = [w.strip() for w in f.readlines() if w]
f.close()


class TagWiki(object):
    def __init__(self):
        self.electrical_links = wikipedia.page("Index_of_electrical_engineering_articles").links
        self.dictionary = gensim.corpora.Dictionary()
        self.lda = None
        self.topdir = WIKI_PATH

    def prepare_dictionary_from_docs(self):
        """
        iterate through the wikipedia docs dir.
        """
        for fn in os.listdir(self.topdir):
            fin = open(os.path.join(self.topdir, fn), 'rb')
            text = fin.read()
            fin.close()
            content = [x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in STOPLIST]
            self.dictionary.add_documents([content])
        self.dictionary.save(DICT_PATH)
        return True

    # Pass 2: Process topics
    def process_topics_from_docs(self):
        if os.path.exists(MODEL_PATH):
            self.lda = gensim.models.ldamodel.LdaModel.load(MODEL_PATH)
        else:
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=None, id2word=self.dictionary, num_topics=30, 
                update_every=30, chunksize=2, passes=10) #, distributed=True)
        f = open(OUTPUT_PATH, "w") 

        for fn in os.listdir(self.topdir):
            try:
                logging.info("processing {0}".format(fn))
                fin = open(os.path.join(self.topdir, fn), 'rb')
                text = fin.read()
                fin.close()
                content = [x for x in gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore") if x not in STOPLIST]
                content_bow = self.dictionary.doc2bow(content)
                self.lda.update([content_bow])
                f.write("{0}::    {1}\n".format(link, sorted(self.lda[new_bag_of_words], key=lambda x: x[1], reverse=True)))
            except:
                logging.info("PROCESSING FAILED!")
                continue
        f.close()
        self.lda.save(MODEL_PATH)
        return True
 
    # Pass 1: Prepare a dictionary
    def prepare_dictionary(self):
        if os.path.exists(DICT_PATH):
            self.dictionary = gensim.corpora.Dictionary.load(DICT_PATH)
            return True

        for link in self.electrical_links:
            try:
                page = wikipedia.page(link)
                logging.info(link)
            except:
                logging.info("failed: {0}".format(link))
                continue
            title = gensim.parsing.preprocess_string(page.title)
            content = gensim.parsing.preprocess_string(page.content)
        
            self.dictionary.add_documents([title, content])
        self.dictionary.save(DICT_PATH)
        return True
    
    # Pass 2: Process topics
    def process_topics(self):
        if os.path.exists(MODEL_PATH):
            self.lda = gensim.models.ldamodel.LdaModel.load(MODEL_PATH)
        else:
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=None, id2word=self.dictionary, num_topics=30, 
                update_every=30, chunksize=2, passes=10) #, distributed=True)
        f = open(OUTPUT_PATH, "w") 
        for link in self.electrical_links:
            try:
                logging.info("processing: {0}".format(link))
                page = wikipedia.page(link)
                title = gensim.parsing.preprocess_string(page.title)
                content = gensim.parsing.preprocess_string(page.content)
            
                title_bow = self.dictionary.doc2bow(title)
                content_bow = self.dictionary.doc2bow(content)
            
                new_bag_of_words = title_bow + content_bow
                self.lda.update([content_bow])
                f.write("{0}::    {1}\n".format(link, sorted(self.lda[new_bag_of_words], key=lambda x: x[1], reverse=True)))
            except:
                logging.info("PROCESSING FAILED!")
                continue
        f.close()
        self.lda.save(MODEL_PATH)
        return True
    
def main():
    start_time = time.time()
    logging.info("START TIME :{0}".format(start_time))
    wiki = TagWiki()
    logging.info("No. of keys at start :{0}".format(wiki.dictionary.keys().__len__()))
    wiki.prepare_dictionary() #_from_docs()
    dict_prepare_time = time.time()
    logging.info("TIME AFTER DICTIONARY PREPARATION :{0}".format(dict_prepare_time))
    wiki.process_topics() #_from_docs()
    first_pass = time.time()
    logging.info("TIME AFTER FIRST PASS :{0}".format(first_pass))
    wiki.process_topics() #_from_docs()
    second_pass = time.time()
    logging.info("TIME AFTER SECOND PASS :{0}".format(second_pass))
    
    
if __name__ == '__main__':
    main()
