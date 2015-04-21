#python svm_train.py {--ID VEC_FILE.pkl ...} --ans TRAIN_ANSWERS --dev DEVNUM --output FILE
# trains an SVM to assign regression weights to tree-wise similarity vectors
# trains/tests on the splits denoted by DEVNUM {1-8}
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
    
if 'output' not in OPTS or 'dev' not in OPTS:
  raise #need someplace to dump the model or this is a waste of time
  
inputlist = [label for label in OPTS if label not in ['ans','output','dev']]
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

print 'X',Xlist.shape
print 'y',ylist.shape

#remove bad training examples
Xlist = Xlist[ylist != -1]
ylist = ylist[ylist != -1]
Xlist = numpy.nan_to_num(Xlist)
devnum = int(OPTS['dev'])
if 'mod' in OPTS:
  #this will ensure that no single domain is left out, but it won't be trivial to eval against the gold standard anymore.
  devX = numpy.delete(Xlist,range(devnum-1,Xlist.shape[0],8),axis=0)
  devY = numpy.delete(ylist,range(devnum-1,ylist.shape[0],8),axis=0)
  testX = Xlist[devnum-1::8]
else:
  #tests on relatively unseen domains
  devX = numpy.concatenate((Xlist[:(devnum-1)*1000],Xlist[devnum*1000:]),axis=0)
  devY = numpy.concatenate((ylist[:(devnum-1)*1000],ylist[devnum*1000:]),axis=0)
  testX = Xlist[(devnum-1)*1000:devnum*1000]
#myobs_scaled = sklearn.preprocessing.scale(myobs_a) #less memory efficient, but centers and scales all features/columns

print 'X',devX.shape
print 'y',devY.shape

#train the SVM regressor based on our training data
model = svm.SVR(kernel='linear')
model.fit(devX, devY)
predictions = model.predict(testX)
predictions = predictions.astype('string',copy=False)
with open(OPTS['output']+OPTS['dev'],'w') as f:
  f.write('\n'.join(predictions) + '\n')
