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
  #ylist = numpy.ravel(ylist) #put ylist in a flattened format
  
for fileid in inputlist:
  #for each composition system, grab the similarity cross-product vector
  print "Incorporating info from %s" % (fileid)
  with open(OPTS[fileid],'rb') as f:
    newfile = pickle.load(f).astype('float64')
    if Xlist == []:
      #if we haven't seen trained output yet, save it
      Xlist = newfile
    else:
      #concatenate each system's training output to the others
      Xlist = numpy.concatenate( (Xlist,newfile), axis=1)

#cnt = 0
#for y in ylist:
  #print type(y), y
#  if y == -1:
    #print 'BOOM'
#    cnt += 1
print 'X',Xlist.shape
#print 'y[0]',ylist[0]
#print 'ycnt', cnt
print 'y',ylist.shape

#remove bad training examples
#keep = numpy.all(numpy.concatenate((ylist != -1, keep),axis=1), axis=1)
Xlist = Xlist[ylist != -1]
ylist = ylist[ylist != -1]
Xlist = numpy.nan_to_num(Xlist)
#keep = numpy.all(numpy.isfinite(Xlist), axis=1)
#Xlist = Xlist[keep]
#ylist = ylist[keep]
#myobs_scaled = sklearn.preprocessing.scale(myobs_a) #less memory efficient, but centers and scales all features/columns

print 'X',Xlist.shape
print 'y',ylist.shape

#train the SVM regressor based on our training data
model = svm.SVR(kernel='linear')
model.fit(Xlist, ylist)

with open(OPTS['output'],'wb') as f:
  pickle.dump(model,f)
