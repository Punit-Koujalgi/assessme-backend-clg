from nltk.corpus import stopwords
import string
import pke 
from AssessMe.utilities.summarize import summarizer
from flashtext import KeywordProcessor
import traceback
from spacy.cli import link
from spacy.util import get_package_path

def get_nouns_multipartite(content,num=15,fitb=False):
    out=[]
    try:
        model_name = "en_core_web_sm"
        package_path = get_package_path(model_name)
        link(model_name, "en", force=True, model_path=package_path)
        out=[]

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','ADJ','NOUN'}
        if fitb:
          pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_weighting(alpha=1.1,
                                    threshold=0.75,
                                    method='average')
        keyphrases = extractor.get_n_best(n=num)
        #print(keyphrases)
        for key in keyphrases:
            out.append(key[0])

    except:
        out = []
        traceback.print_exc()

    return out

def get_keywords(originaltext,num):
  # summarytext = summarizer(originaltext)
  #print(summarytext)
  keywords = get_nouns_multipartite(originaltext,num)
  #print ("keywords unsummarized: ",keywords)
  # keyword_processor = KeywordProcessor()
  # for keyword in keywords:
  #   keyword_processor.add_keyword(keyword)

  # keywords_found = keyword_processor.extract_keywords(summarytext)
  # keywords_found = list(set(keywords_found))
  # #print ("keywords_found in summarized: ",keywords_found)

  # important_keywords =[]
  # for keyword in keywords:
  #   if keyword in keywords_found:
  #     important_keywords.append(keyword)

  # return important_keywords,summarytext
  return keywords,""

# text = """A Lion lay asleep in the forest, his great head resting on his paws. A timid little Mouse came upon him unexpectedly, and in her fright and haste to
# get away, ran across the Lion's nose. Roused from his nap, the Lion laid his huge paw angrily on the tiny creature to kill her.  "Spare me!" begged
# the poor Mouse. "Please let me go and some day I will surely repay you."  The Lion was much amused to think that a Mouse could ever help him. But he
# was generous and finally let the Mouse go.  Some days later, while stalking his prey in the forest, the Lion was caught in the toils of a hunter's
# net. Unable to free himself, he filled the forest with his angry roaring. The Mouse knew the voice and quickly found the Lion struggling in the net.
# Running to one of the great ropes that bound him, she gnawed it until it parted, and soon the Lion was free.  "You laughed when I said I would repay
# you," said the Mouse. "Now you see that even a Mouse can help a Lion." """

# print(get_keywords(text))