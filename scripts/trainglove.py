# python egraphTree.py train.sents output.pkl
import cPickle as pickle
import egraphTree as eT
import sys

#load vector dict
with open('/home/corpora/original/english/glove.42B.300d.txt','r') as f:
  modellist = f.readlines()

model = {}
for line in modellist:
  #build model
  sline = line.strip().split()
  model[sline[0]] = sline[1:]
  
#load data for treeing
with open(sys.argv[1],'r') as f:
  trainsents = f.readlines()

#build a list of paired training trees
traintrees = []
for pair in trainsents:
  treepair = []
  for pairelem in pair.strip().split('\t'):
    nodes = eT.compose(pairelem)
    mtree = eT.nodes2Tree(nodes)
    mtree.annotateVectors(model)
    treepair.append(mtree)
  traintrees.append(treepair)

#write the trained output trees
with open(sys.argv[2],'wb') as f:
  pickle.dump(traintrees,f)
