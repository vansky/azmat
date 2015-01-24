#python svm_coefs.py --model SVM_MODEL.pkl > tmp
# outputs the trained weights from a trained SVM regressor to stdout

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

if 'model' not in OPTS:
  raise
    
with open(OPTS['model'],'rb') as f:
  model = pickle.load(f)
  
#print model.coef_.shape
sys.stderr.write(str(model.coef_.shape)+'\n')
  
#iterate over each weight and print it to stdout
for dim in model.coef_.ravel():
  print dim
