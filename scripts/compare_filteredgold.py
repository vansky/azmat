#python compare_filteredgold.py --gold GFILE --test TFILE --goldout NEWGFILE --testout NEWTFILE
# compares line-delimited GFILE and line-delimited TFILE removing all blank lines in GFILE along with the corresponding lines in TFILE
# outputs the resulting GFILE and TFILE to NEWGFILE and NEWTFILE respectively
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
    
if 'gold' not in OPTS or 'test' not in OPTS or 'goldout' not in OPTS or 'testout' not in OPTS:
  raise #need someplace to dump the model or this is a waste of time

with open(OPTS['gold'], 'r') as f:
  gold = [l.strip() for l in f.readlines()]

with open(OPTS['test'], 'r') as f:
  test = [l.strip() for l in f.readlines()]

goldout = []
testout = []
  
for li, l in enumerate(gold):
  if l != '':
    #don't skip this line
    goldout.append(gold[li])
    testout.append(test[li])

with open(OPTS['goldout'],'w') as f:
  f.write('\n'.join(goldout)+'\n')

with open(OPTS['testout'],'w') as f:
  f.write('\n'.join(testout)+'\n')
