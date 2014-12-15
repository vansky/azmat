#python svm_test.py --model SVM_MODEL.pkl {--test EVAL_FILE.pkl ... } --output FILE
# uses a trained SVM regressor to output similarity of sentence similarity vectors in EVAL_FILEs
# IMPORTANT: eval systems must be given in same order as during training
# model predictions are output to the FILE specified by --output

import numpy
import cPickle as pickle
from sklearn import svm
import sys

testlist = []
OPTS = {}
for aix in range(1,len(sys.argv)):
  if len(sys.argv[aix]) < 2 or sys.argv[aix][:2] != '--':
    #filename or malformed arg
    continue
  elif aix < len(sys.argv) - 1 and len(sys.argv[aix+1]) > 2 and sys.argv[aix+1][:2] == '--':
    #missing filename, so simple arg
    OPTS[sys.argv[aix][2:]] = True
  elif sys.argv[aix][2:] == 'test':
    testlist.append(sys.argv[aix+1])
  else:
    OPTS[sys.argv[aix][2:]] = sys.argv[aix+1]

if 'output' not in OPTS:
  raise #need someplace to dump the model or this is a waste of time

with open(OPTS['model'],'wb') as f:
  model = pickle.load(f)

Xlist = []
for testfile in testlist:
  #for each composition system, grab the similarity cross-product vector
  with open(infile,'rb') as f:
    newfile = pickle.load(f)
    Xlist.append(newfile)

#get everything in array format
X = numpy.array(Xlist)

#train the SVM regressor based on our training data
predictions = model.predict(X)

with open(OPTS['output'],'wb') as f:
  pickle.dump(predictions,f)
