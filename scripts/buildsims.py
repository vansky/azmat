#python buildsims.py sims output.pkl
#compiles predictions/similarity judgements into a pickle file

import cPickle as pickle
import numpy
import sys

with open(sys.argv[1],'r') as f:
  sims = f.readlines()

simpick = []
for line in sims:
  simpick.append(line.strip())

simpick = numpy.array(simpick)
with open(sys.argv[2],'wb') as f:
  pickle.dump(simpick,f)
