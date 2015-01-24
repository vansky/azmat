#python capsim.py sim.pkl [--text] > output
#strips capped value (0-MAX) predictions/similarity judgements from a pickle file
#if --text arg is given, reads from a text file instead of a pickle file

from __future__ import division
import cPickle as pickle
import numpy
import sys

MAX = 5.0

if len(sys.argv) > 2 and sys.argv[2] == '--text':
  sims = numpy.loadtxt(sys.argv[1])
else:
  with open(sys.argv[1],'rb') as f:
    sims = pickle.load(f)

for r in sims:
  if r < 0:
    print( 0.0 )
  elif r > MAX:
    print( MAX )
  else:
    print(r)
