import gensim
import wikipedia
from gensim.parsing.preprocessing import STOPWORDS

from tag_wikipedia_articles import *
from topic_modeling import *

topic_name = 'Harmonic oscillator'

WIKI_PATH = '/home/ubuntu/Wiki/en/20150805/enwiki-20150805-pages-articles.xml.bz2'

def test_wiki_page():
    page = wikipedia.page(topic_name)
    
    print 'Name:', page.title
    print 'Content:', page.content[:100]
    
    print 'CLEANED CONTENT...'
    print 'TITLE:', gensim.parsing.preprocess_string(page.title)
    print 'CONTENT:', gensim.parsing.preprocess_string(page.content)

def test_topic_modelling():
    stream = iter_wiki(WIKI_PATH)
    for title, tokens in itertools.islice(iter_wiki(WIKI_PATH), 8):
        print title, tokens[:10]  # print the article title and its first ten tokens

# create dictionary
doc_stream = (tokens for _, tokens in iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2'))
id2word_wiki = gensim.corpora.Dictionary(doc_stream)
print(id2word_wiki)

# filtering out extremes - data preparation
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
print(id2word_wiki)


#Vectorization
doc = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."
bow = id2word_wiki.doc2bow(tokenize(doc))
print(bow)


# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus('./data/simplewiki-20140623-pages-articles.xml.bz2', id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream


# what is the most common word in that first article?
most_index, most_count = max(vector, key=lambda (word_index, count): count)
print(id2word_wiki[most_index], most_count)

gensim.corpora.MmCorpus.serialize('./data/wiki_bow.mm', wiki_corpus)

mm_corpus = gensim.corpora.MmCorpus('./data/wiki_bow.mm')

# Semantic transformations
clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)

# print least few important topics
print lda_model.print_topics(-1)

tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)

#The TFIDF transformation only modifies feature weights of each word. Its input and output dimensionality are identical (=the dictionary size).
lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)



tfidf_corpus = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm')
# `tfidf_corpus` is now exactly the same as `tfidf_model[wiki_corpus]`
print(tfidf_corpus)

lsi_corpus = gensim.corpora.MmCorpus('./data/wiki_lsa.mm')
# and `lsi_corpus` now equals `lsi_model[tfidf_model[wiki_corpus]]` = `lsi_model[tfidf_corpus]`
print(lsi_corpus)


# Transforming unseen documents
text = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."

# transform text into the bag-of-words space
bow_vector = id2word_wiki.doc2bow(tokenize(text))
print([(id2word_wiki[id], count) for id, count in bow_vector])

# transform into LDA space
lda_vector = lda_model[bow_vector]
print(lda_vector)
# print the document's single most prominent LDA topic
print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))


# transform into LSI space
lsi_vector = lsi_model[tfidf_model[bow_vector]]
print(lsi_vector)
# print the document's single most prominent LSI topic (not interpretable like LDA!)
print(lsi_model.print_topic(max(lsi_vector, key=lambda item: abs(item[1]))[0]))

# store all trained models to disk
lda_model.save('./data/lda_wiki.model')
lsi_model.save('./data/lsi_wiki.model')
tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save('./data/wiki.dictionary')


# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load('./data/lda_wiki.model')



#Evaluation
# select top 50 words for each of the 20 LDA topics
top_words = [[word for _, word in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
print(top_words)

# get all top 50 words in all 20 topics, as one large set
all_words = set(itertools.chain.from_iterable(top_words))

print("Can you spot the misplaced word in each topic?")

# for each topic, replace a word at a different index, to make it more interesting
replace_index = np.random.randint(0, 10, lda_model.num_topics)

replacements = []
for topicno, words in enumerate(top_words):
    other_words = all_words.difference(words)
    replacement = np.random.choice(list(other_words))
    replacements.append((words[replace_index[topicno]], replacement))
    words[replace_index[topicno]] = replacement
    print("%i: %s" % (topicno, ' '.join(words[:10])))


print("Actual replacements were:")
print(list(enumerate(replacements)))

# evaluate on 1k documents **not** used in LDA training
doc_stream = (tokens for _, tokens in iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2'))  # generator
test_docs = list(itertools.islice(doc_stream, 8000, 9000))


def intra_inter(model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    part1 = [model[id2word_wiki.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[id2word_wiki.doc2bow(tokens[len(tokens) / 2 :])] for tokens in test_docs]
    
    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))
    
    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))



print("LDA results:")
intra_inter(lda_model, test_docs)

print("LSI results:")
intra_inter(lsi_model, test_docs)