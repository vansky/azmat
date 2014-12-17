from copy import deepcopy
import egraphTree as eT
import cPickle as pickle
import numpy as np

def randvec(l = 10):
  return( np.random.random((l,)) )

def addVectors(intree):
  intree.vector = randvec()
  for ch in intree.ch:
    addVectors(ch)

def duptree(intree):
  addVectors(intree)
  return( deepcopy(intree) )

sentence = "00/1/03 01/0/N-lI:April 02/0/D-aN-lI:'s 02/1/03 02/2/01 03/0/N-aD-lI:flowers 04/0/V-aN-b{A-aN}-lI:are 04/=/03 05/0/D-lI:the 05/1/03 06/0/N-aD-lI:Sweet 06/=/07 07/0/N-aD-lI:Pea 08/0/X-cX-dX-lI:and 07/&/08 09/&/08 08/=/03 09/0/N-aD-lI:Daisy 10/0/.-lI:. 10/1/03"
mtree = eT.nodes2Tree(eT.compose(sentence))

trainout = []
for i in range(1000):
  trainelem = np.array( [duptree(mtree), duptree(mtree)] )
  trainout.append(trainelem)

trainout = np.array(trainout)

with open('dummytrees.pkl','wb') as f:
  pickle.dump(trainout,f)

sims = randvec(1000)
with open('dummysims.pkl','wb') as f:
  pickle.dump(sims,f)
