#python svm_train.py {--ID VEC_FILE.pkl ...} --ans TRAIN_ANSWERS --output FILE
# trains an SVM to assign regression weights to tree-wise similarity vectors
# TRAIN_ANSWERS needs to be an X-by-1 numpy array of similarity training answers
# model is output to the FILE specified by --output

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
#  elif sys.argv[aix][2:] == 'input':
#    inputlist.append(sys.argv[aix+1])
  else:
    OPTS[sys.argv[aix][2:]] = sys.argv[aix+1]

if 'output' not in OPTS:
  raise #need someplace to dump the model or this is a waste of time

inputlist = [label for label in OPTS if label not in ['ans','output']]
inputlist = sorted(inputlist) #arrange systems alphabetically according to cli identifier

Xlist = []
ylist = None
with open(OPTS['ans'],'rb') as f:
  #snag the training answers
  ylist = pickle.load(f)
  ylist = numpy.ravel(ylist) #put ylist in a flattened format

for infile in inputlist:
  #for each composition system, grab the similarity cross-product vector
  with open(infile,'rb') as f:
    newfile = pickle.load(f)
    if Xlist == []:
      #if we haven't seen trained output yet, save it
      Xlist = newfile
    else:
      #concatenate each system's training output to the others
      Xlist = numpy.concatenate( (Xlist,newfile), axis=1)

#myobs_scaled = sklearn.preprocessing.scale(myobs_a) #less memory efficient, but centers and scales all features/columns

#train the SVM regressor based on our training data
model = svm.SVR(kernel='linear')
model.fit(X, y)

with open(OPTS['output'],'wb') as f:
  pickle.dump(model,f)
