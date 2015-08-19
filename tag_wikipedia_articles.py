import os
import wikipedia
import gensim
import logging

logging.basicConfig(filename='wiki.log', level=logging.INFO)

DICT_PATH = "/tmp/wiki.dict"
MODEL_PATH = "/tmp/wiki.lda"
OUTPUT_PATH = "/tmp/output.txt"


class TagWiki(object):
    def __init__(self):
        self.electrical_links = wikipedia.page("Index_of_electrical_engineering_articles").links
        self.dictionary = gensim.corpora.Dictionary()
        self.lda = None

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
                update_every=1, chunksize=1, passes=2)
        f = open(OUTPUT_PATH, "w") 
        for link in self.electrical_links:
            try:
                page = wikipedia.page(link)
                title = gensim.parsing.preprocess_string(page.title)
                content = gensim.parsing.preprocess_string(page.content)
            
                title_bow = self.dictionary.doc2bow(title)
                content_bow = self.dictionary.doc2bow(content)
            
                new_bag_of_words = title_bow + content_bow
                self.lda.update([content_bow])
                f.write("{0}::    {1}\n".format(link, self.lda.print_topics(5)))
                logging.info("RESULT: {0}: {1}".format(link, self.lda[new_bag_of_words]))
            except:
                logging.info("PROCESSING FAILED!")
                continue
        f.close()
        self.lda.save(MODEL_PATH)
        return True
    
def main():
    wiki = TagWiki()
    logging.info("No. of keys at start :{0}".format(wiki.dictionary.keys().__len__()))
    wiki.prepare_dictionary()
    logging.info("No. of keys after addition of data {0}".format(wiki.dictionary.keys().__len__()))
    wiki.process_topics()
    logging.info("done : {0}".format(wiki.dictionary.__sizeof__()))
    
    
if __name__ == '__main__':
    main()
