import gensim
import wikipedia
from gensim.parsing.preprocessing import STOPWORDS

from tag_wikipedia_articles import *

topic_name = 'Harmonic oscillator'


def test_wiki_page():
    page = wikipedia.page(topic_name)
    
    print 'Name:', page.title
    print 'Content:', page.content[:100]
    
    print 'CLEANED CONTENT...'
    print 'TITLE:', gensim.parsing.preprocess_string(page.title)
    print 'CONTENT:', gensim.parsing.preprocess_string(page.content)

def test_topic_modelling():
    stream = iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2')
    for title, tokens in itertools.islice(iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2'), 8):
            print title, tokens[:10]  # print the article title and its first ten tokens
