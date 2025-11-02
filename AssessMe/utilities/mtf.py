import nltk
import pke
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
from AssessMe.utilities.wsd import get_best_sense


def tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_keywords(text):
    out=[]
    try:
        # extractor = pke.unsupervised.MultipartiteRank()
        extractor = pke.unsupervised.YAKE()
        extractor.load_document(input=text)
        # pos = {'VERB', 'ADJ', 'NOUN'}
        pos ={'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(n=1,pos=pos, stoplist=stoplist)

        extractor.candidate_weighting(window=3,
                                      stoplist=stoplist,
                                      use_stems=False)

        keyphrases = extractor.get_n_best(n=15)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []

    return out

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=False)
        keyword_sentences[key] = values
    return keyword_sentences

def get_mtf(context):
    keywords=get_keywords(context)
    sentences=tokenize_sentences(context)
    keyword_sentence_mapping=get_sentences_for_keyword(keywords,sentences)
    keys=[]
    defs=[]
    for keyword in keywords[:7]:
        syns=wn.synsets(keyword,'n')
        if len(syns)==0 or len(syns)>6 or keyword in keys: continue
        syn=get_best_sense(keyword,keyword_sentence_mapping[keyword][0],syns)
        keys.append(keyword)
        defs.append(syn.definition())
    return keys,defs