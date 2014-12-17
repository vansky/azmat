#python svm_test.py --model SVM_MODEL.pkl {--ID EVAL_FILE.pkl ... } --output FILE
# uses a trained SVM regressor to output similarity of sentence similarity vectors in EVAL_FILEs
# IMPORTANT: eval systems must use the same ID as during testing (or at least be in the same alphabetical position)
# model predictions are output to the FILE specified by --output

import numpy
import cPickle as pickle
from sklearn import svm
import sys

OPTS = {}
for aix in range(1,len(sys.argv)):
  if len(sys.argv[aix]) < 2 or sys.argv[aix][:2] != '--':
    #filename or malformed arg
    continue
  elif aix < len(sys.argv) - 1 and len(sys.argv[aix+1]) > 2 and sys.argv[aix+1][:2] == '--':
    #missing filename, so simple arg
    OPTS[sys.argv[aix][2:]] = True
  else:
    OPTS[sys.argv[aix][2:]] = sys.argv[aix+1]

if 'output' not in OPTS:
  raise #need someplace to dump the output or this is a waste of time

testlist = [label for label in OPTS if label not in ['model','output']]
testlist = sorted(testlist) #arrange systems alphabetically according to cli identifier

with open(OPTS['model'],'rb') as f:
  model = pickle.load(f)

Xlist = []

for fileid in testlist:
  #for each composition system, grab the similarity cross-product vector
  with open(OPTS[fileid],'rb') as f:
    newfile = pickle.load(f)
    if Xlist == []:
      #if we haven't seen trained output yet, save it
      Xlist = newfile
    else:
      #concatenate each system's training output to the others
      Xlist = numpy.concatenate( (Xlist,newfile), axis=1)

#train the SVM regressor based on our training data
predictions = model.predict(Xlist)

with open(OPTS['output'],'wb') as f:
  pickle.dump(predictions,f)
