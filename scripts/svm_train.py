#python svm_train.py {--input VEC_FILE.pkl ...} --output FILE
# trains an SVM to assign regression weights to tree-wise similarity vectors
# model is output to the FILE specified by --output

import numpy
import cPickle as pickle
from sklearn import svm
import sys

#RANDOM_SEED = 37 #None yields random initialization

inputlist = []
OPTS = {}
for aix in range(1,len(sys.argv)):
  if len(sys.argv[aix]) < 2 or sys.argv[aix][:2] != '--':
    #filename or malformed arg
    continue
  elif aix < len(sys.argv) - 1 and len(sys.argv[aix+1]) > 2 and sys.argv[aix+1][:2] == '--':
    #missing filename, so simple arg
    OPTS[sys.argv[aix][2:]] = True
  elif sys.argv[aix][2:] == 'input':
    inputlist.append(sys.argv[aix+1])
  else:
    OPTS[sys.argv[aix][2:]] = sys.argv[aix+1]

if 'output' not in OPTS:
  raise #need someplace to dump the model or this is a waste of time
    
Xlist = []
ylist = []
for infile in inputlist:
  #for each composition system, grab the similarity cross-product vector
  with open(infile,'rb') as f:
    newfile = pickle.load(f)
    Xlist.append(newfile['vec'])
    if ylist == []:
      #if we haven't seen the training answers yet, snag them
      ylist = newfile['ans']

#get everything in array format
X = numpy.array(Xlist)
y = numpy.array(ylist)

#myobs_scaled = sklearn.preprocessing.scale(myobs_a) #less memory efficient, but centers and scales all features/columns

#train the SVM regressor based on our training data
model = svm.SVR(kernel='linear')
model.fit(X, y)

with open(OPTS['output'],'wb') as f:
  pickle.dump(model,f)
