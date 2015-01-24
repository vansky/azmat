#python abssim.py sim.pkl [--text] > output
#strips absolute values scales (0-MAX) valued predictions/similarity judgements from a pickle file
#if --text arg is given, reads from a text file instead of a pickle file

from __future__ import division
import cPickle as pickle
import numpy
import sys

MAX = 5

if len(sys.argv) > 2 and sys.argv[2] == '--text':
  sims = numpy.loadtxt(sys.argv[1])  
else:
  with open(sys.argv[1],'rb') as f:
    sims = pickle.load(f)

neg = 0
pos = 0
tot = 0
for r in sims:
  if r < 0:
    neg += 1
  elif r > 5:
    pos += 1
  tot += 1
sys.stderr.write('neg: '+str(neg)+' pos: '+str(pos)+' tot: '+str(tot)+'\n')
sys.stderr.write('min: '+str(numpy.min(sims))+' max: '+str(numpy.max(sims))+'\n')
  
sims = abs(sims)
sims -= numpy.min(sims) #shift negatives up to zero; shift positives down to zero
divisor = numpy.max(sims) / MAX
sims /= divisor

for r in sims:
  print(r)
