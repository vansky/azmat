#python pullsim.py sim.pkl > output
#strips predictions/similarity judgements from a pickle file

import cPickle as pickle
import numpy
import sys

with open(sys.argv[1],'rb') as f:
  sims = pickle.load(f)

for r in sims:
  print(r)
