import logging
import itertools

import numpy as np
import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


class WikiCorpus(object):
    def __init__(self, dump_file, dictionary=None, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(self.iter_wiki(), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

    def tokenize(self, text):
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]

    def iter_wiki(self):
        """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
        for title, text, pageid in _extract_pages(smart_open(self.dump_file)):
            text = filter_wiki(text)
            tokens = self.tokenize(text)
            if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
                continue  # ignore short articles and various meta-articles
            yield title, tokens


def main(wiki_path):
    # create a stream of bag-of-words vectors
    wiki_corpus = WikiCorpus(wiki_path)
    vector = next(iter(wiki_corpus))
    print(vector)  # print the first vector in the stream


if __name__ == '__main__':
    wiki_path = sys.argv[1]
    main(wiki_path)
