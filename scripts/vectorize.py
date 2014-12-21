# An API to get a similarity score between two trees
# Usage: getSim(treeA,treeB)
# PRE: treeA/treeB - binarized trees with vectors representing each node
# POST: a vector representation of the similarity of each node to each other node

#If run using CLI:
#python vectorize.py TREEFILE OUTFILE
# reads in TREEFILE, which contains an X-by-2 numpy array (or list) of treeA,treeB pairs for each training pair
# outputs a numpy array with a similarity vector on each row for each training example

from egraphTree import *
import numpy as np
import cPickle as pickle
from scipy import linalg, dot
import sys

######
# Helper junk for dev
######

def printDepth(treeA):
  #print the depth of each node in treeA
  for ch in treeA.ch:
    printDepth(ch)
  print treeA.c, treeA.depth
def annotateVec(treeA):
  #annotate a tree with random vectors
  treeA.vector = np.random.uniform(0,1,size=10)
  for ch in treeA.ch:
    annotateVec(ch)

######
######

def getNodes(treeA):
  #get all the nodes and associated depths from treeA
  output = []
  #add the current node
  output.append( (treeA.vector,treeA.depth) )
  for ch in treeA.ch:
    #for each child node, add its output
    output += getNodes(ch)
  return ( output )

def similarity(a,b):
  #return the cosine similarity of two 1-D arrays
  a = np.array(a)
  b = np.array(b)
  return (dot(a,b)/linalg.norm(a)/linalg.norm(b) )

def fillout(vec,length):
  #fill out a vector to given length
  output = []
  maxlen = len(vec)
  for l in range(int(length)):
    #repeatedly iterate over vec until desired length
    output.append(vec[l % maxlen])
  return( sorted(output) )

def getSim(treeA,treeB):
  #Get the similarity vector for treeA and treeB
  if treeA.c == '' or treeB.c == '':
    nodelistA = []
    nodelistB = []
  else:
    nodelistA = getNodes(treeA)
    nodelistB = getNodes(treeB)

  simdict = {}
  simvec = []
  #find the cross product similarity scores
  for nodea in nodelistA:
    for nodeb in nodelistB:
      #key the similarity score by the depth pair
      if (nodea[1],nodeb[1]) not in simdict:
        simdict[(nodea[1],nodeb[1])] = []
      simdict[(nodea[1],nodeb[1])].append( similarity(nodea[0],nodeb[0]) )
  #unify microdiffs so 0x0 is always the same length
  for key in simdict:
    length = 50.0/(2**key[0]) * 50.0/(2**key[1])
    simdict[key] = fillout(sorted(simdict[key]), length)

  #find all the keys that every sentence should have
  keylist = sorted([(x,y) for x in range(50) for y in range(50)])

  for key in keylist:
    #for each key, find the length it should be (assuming len(50) sentences)
    length = 50.0/(2**key[0]) * 50.0/(2**key[1])

    if key in simdict:
      #if we've seen that kind of depth pairing:
      #  add it to our vector after expanding it to the requisite length
      simvec += fillout(sorted(simdict[key]), length)
    else:
      #if we've not seen that kind of depth pairing:
      #  add a null vector with the right length
      simvec += fillout([0], length)
  return( np.array(simvec) )

def buildSimMatrix(fileHandle):
  with open(fileHandle,'rb') as f:
    treefile = pickle.load(f)
  matrix = []
  for row in treefile:
    if len(row) < 2:
      row = np.array([tree.Tree(),tree.Tree()])
    if type(row) == type([]):
      row = np.array(row)
    if matrix == []:
      matrix = getSim(row[0],row[1]).reshape(1,-1)
    else:
      matrix = np.concatenate( (matrix, getSim(row[0],row[1]).reshape(1,-1) ), axis= 0)
    #print matrix.shape
  return( matrix )

if __name__ == '__main__':
  output = buildSimMatrix(sys.argv[1])
  print output.shape
  with open(sys.argv[2],'wb') as f:
    pickle.dump(output,f)
