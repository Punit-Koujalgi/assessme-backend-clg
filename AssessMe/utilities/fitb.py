from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from AssessMe.utilities.keywords import get_nouns_multipartite
import re
from pprint import pprint

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 35]
    return sentences

def get_fill_in_the_blanks(text):
    keywords = get_nouns_multipartite(text,40,True)
    sentences=tokenize_sentences(text)
    keyword_processor=KeywordProcessor()
    keyword_sentences={}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    # pprint(keywords)
    ques=[]
    processed=[]
    count=0
    for key in keyword_sentences:
        if len(keyword_sentences[key])>0:
            q={}
            sent = keyword_sentences[key][0]
            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            if (keyword_sentences[key][0] not in processed) and no_of_replacements<2:
                processed.append(keyword_sentences[key][0])
                q['id']=count
                q['question']=line
                q['answer']=key
                count+=1
                ques.append(q)
    return ques
