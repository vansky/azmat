# python trainglove.py train.sents output.pkl
import numpy as np
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
with open(sys.argv[1],'rb') as f:
  traintrees = pickle.load(f)

#use a list of paired training trees
for pair in traintrees:
  for tree in pair:
    tree.annotateVectors(model)

#write the trained output trees
with open(sys.argv[2],'wb') as f:
  pickle.dump(traintrees,f)
