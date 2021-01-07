# Import Dataset
import re
import gensim
from Gensim_Model import Gensim_Model
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
# spacy for lemmatization
import spacy


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def openapi_preprocess(data):
    # Remove html tags
    data = [re.sub('<[^>]*>', '', sent) for sent in data]
    data = [re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', sent) for sent in data]
    return data

def generate_bigrams_and_trigrams(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod


def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(brigrams, texts):
    return [brigrams[doc] for doc in texts]

def make_trigrams(trigrams, bigrams, texts):
    return [trigrams[bigrams[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []

    for sent in texts:
        doc = nlp(" ".join(sent))
        if(allowed_postags != None):
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        else:
            texts_out.append([token.lemma_ for token in doc])

    return texts_out

def execute_preprocessing(data_list, post=['NOUN', 'ADJ', 'VERB', 'ADV']):
    data_list = openapi_preprocess(data_list)
    data_words = list(sent_to_words(data_list))
    bigrams, trigrams = generate_bigrams_and_trigrams(data_words)
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    #data_words_bigrams = make_bigrams(bigrams, data_words_nostops)
    data_words_trigrams = make_trigrams(trigrams, bigrams, data_words_nostops)

    # Do lemmatization
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=post)

    return data_lemmatized


def execute_preprocessing_and_update_info(data_list, data_info, post=['NOUN', 'ADJ', 'VERB', 'ADV']):
    data_lemmatized = execute_preprocessing(data_list, post)

    lem_left = []
    new_info = {}
    id = 0
    #Remove empty documents
    for i in range(len(data_lemmatized)):
        if(data_lemmatized[i] != [] and not data_lemmatized[i] in lem_left):
            lem_left.append(data_lemmatized[i])
            new_info[id] = data_info[i]
            id += 1

    return lem_left, new_info
