# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# random
import random
from random import shuffle

# stop words from nltk
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))

class LabeledIndividualSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    line = line.lower()
                    tokens = utils.to_unicode(line).split()
                    tokens = [w for w in tokens if not w in stopset]
                    yield LabeledSentence(tokens, [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    line = line.lower() # change case for better results
                    tokens = utils.to_unicode(line).split() # LabeledSentence accepts only unicode tokens
                    tokens = [w for w in tokens if not w in stopset] # remove stopwords
                    self.sentences.append(LabeledSentence(tokens, [prefix + '_%s' % item_no]))
        return self.sentences
    
    def getTag_words(self,words):
        return [s for s in self.to_array() if ' '.join(s.words)==words][0].tags[0]
    
    def getWords_tag(self,tag):
        return ' '.join([s for s in self.to_array() if s.tags[0]==tag][0].words)


sources = {'train-blocked.txt':'TRAIN_BL', 'train-nonblocked.txt':'TRAIN_NBL'}
sentences = LabeledIndividualSentence(sources)
alldocs = sentences.to_array()
doc_list = alldocs[:]  # for reshuffling per pass

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(alldocs)
for interval in range(20):
    shuffle(doc_list) # reshuffling for better results
    model.train(doc_list)

wrd = "fang lizhi"
print("Query word : ",wrd, "\n")
sims = model.docvecs.most_similar(sentences.getTag_words(wrd), topn=4)
#print(sims)
tags_sims = [s[0] for s in sims]
for t in tags_sims:
    print(sentences.getWords_tag(t),'\n')