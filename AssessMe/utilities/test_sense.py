from AssessMe.utilities.mtf import get_mtf
from nltk.corpus import wordnet as wn
from AssessMe.utilities.wsd import get_best_sense
from pprint import pprint

sentence = '''Once upon a time, there lived a lion in the dense Amazon rainforest. While he was sleeping by resting his big head on his paws, a tiny little mouse unexpectedly crossed by and ran across the lion’s nose in haste. This woke up the lion and he laid his huge paw angrily on the tiny mouse to kill her.

The poor mouse begged the lion to spare her this time and she would pay him back on some other day. Hearing this, the lion was amused and wondered how could such a tiny creature ever help him. But he was in a good mood and in his generosity he finally let the mouse go.

A few days later, a hunter set a trap for the lion while the big animal was stalking for prey in the forest. Caught in the toils of a hunter’s net, the lion found it difficult to free himself and roared loudly in anger.

As the mouse was passing by, she heard the roar and found the lion struggling hard to free himself from the hunter’s net. The little creature quickly ran towards the lion’s trap that bound him and she gnawed the net with her sharp teeth until the net tore apart. Slowly she made a big hole in the net and soon the lion was able to free himself from the hunter’s trap.

The lion thanked the little mouse for her help and the mouse reminded him that she had finally repaid the lion for sparing her life before. Thereafter, the lion and the mouse became good friends and lived happily in the forest.'''

context=sentence.replace("\n","")
keys,defs=get_mtf(context)
pprint(keys)
pprint(defs)
# word="Cricket"

# syns=wn.synsets(word)
# syn=get_best_sense(word,sentence,syns)

# print(syn.definition())


