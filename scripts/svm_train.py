#python svm_train_coefs.py {--ID VEC_FILE.pkl ...} --ans TRAIN_ANSWERS --output FILE
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

if 'dev' in OPTS:
  DEV = True
else:
  DEV = False

if 'leafonly' in OPTS:
  LEAF = True
  COMPOS = False
  CROSS = False
elif 'composeonly' in OPTS:
  LEAF = False
  COMPOS = True
  CROSS = False
elif 'crossonly' in OPTS:
  LEAF = False
  COMPOS = False
  CROSS = True
elif 'leafcompose' in OPTS:
  LEAF = True
  COMPOS = True
  CROSS = False
elif 'leafcross' in OPTS:
  LEAF = True
  COMPOS = False
  CROSS = True
elif 'composecross' in OPTS:
  LEAF = False
  COMPOS = True
  CROSS = True
else:
  LEAF = True
  COMPOS = True
  CROSS = True

if LEAF:
  print 'leaf/leaf'
if COMPOS:
  print 'nonleaf/nonleaf'
if CROSS:
  print 'leaf/nonleaf'
  
inputlist = [label for label in OPTS if label not in ['ans','output','dev','leafonly','crossonly','composeonly','leafcompose','leafcross','composecross']]
inputlist = sorted(inputlist) #arrange systems alphabetically according to cli identifier

Xlist = []
ylist = None
with open(OPTS['ans'],'rb') as f:
  #snag the training answers
  ylist = pickle.load(f)
  #ylist = numpy.ravel(ylist) #put ylist in a flattened format

print 'Generating Mask'
keylist = sorted([(x,y) for x in range(50) for y in range(50)])
totallength = 0
for key in keylist:
  totallength += int(50.0/(2**key[0]) * 50.0/(2**key[1]))
modelmask = numpy.ones((1,totallength),dtype=bool)

index = 0
if not (LEAF and CROSS and COMPOS):
  for key in keylist:
    subvectorlength = int(50.0/(2**key[0]) * 50.0/(2**key[1]))
    if key == (0,0):
      #leaf/leaf comparison
      if not LEAF:
        #wipe all weights associated with leaf/leaf similarity
        for weightix in range(index,index+subvectorlength):
          modelmask[0,weightix] = False
    elif 0 in key:
      #leaf/nonleaf comparison
      if not CROSS:
        #wipe all weights associated with leaf/nonleaf similarity
        for weightix in range(index,index+subvectorlength):
          modelmask[0,weightix] = False
    else:
      #nonleaf/nonleaf similarity
      if not COMPOS:
        #wipe all weights associated with leaf/nonleaf similarity
        for weightix in range(index,index+subvectorlength):
          modelmask[0,weightix] = False
    #when done wiping or skipping, move to the next subvector
    index += subvectorlength

modelmask = modelmask.ravel()
print modelmask.shape
print 'Mask Generated'
  
for fileid in inputlist:
  #for each composition system, grab the similarity cross-product vector
  print "Incorporating info from %s" % (fileid)
  with open(OPTS[fileid],'rb') as f:
    newfile = pickle.load(f).astype('float64')
    print newfile.shape
    if 'surf' not in fileid:
      newfile = newfile[:,modelmask]
    print 'masked:', newfile.shape
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
if DEV:
  Xlist = Xlist[:-1000]
  ylist = ylist[:-1000]

#myobs_scaled = sklearn.preprocessing.scale(myobs_a) #less memory efficient, but centers and scales all features/columns

print 'X',Xlist.shape
print 'y',ylist.shape

#train the SVM regressor based on our training data
model = svm.SVR(kernel='linear')
model.fit(Xlist, ylist)

with open(OPTS['output'],'wb') as f:
  pickle.dump(model,f)
