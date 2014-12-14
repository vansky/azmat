from collections import defaultdict
import copy
from operator import itemgetter
import tree
import pdb

def parseEgraph(egraph):
    egraph = egraph.strip().split()
    localVocab = defaultdict(list)
    relation = defaultdict(list)
    for item in egraph:
        if ':' in item:
            localVocab[item[:2]] = item[item.index(':')+1:] #e.g., "03":"flowers"
        else:
            item = item.split('/')
            relation[item[0]].append((item[2],item[1])) #e.g., "10":[("03","1")], meaning there is an arc between 10 and 3, whose label is 1
    for item in relation:
        #sort arcs
        relation[item].sort(key = lambda x : (x[1], abs(int(item) - int(x[0]))), reverse = True)
    #add root label for '00' node
    localVocab['#ROOT'] = '00'
    return localVocab, relation
    
def compose(egraph):
    localVocab, relation = parseEgraph(egraph)
    #print relation
    startingNode = relation['00'][0][0]
    curW, steps, _= recurCompose(localVocab, relation, startingNode)
    stepDict = steps # del
    #print steps
    return steps

def nodes2Tree(nodes):
    #convert terminals to depth 0
    for node in nodes:
        if len(node) == 2: #e.g. ("Pea","07")
            nodes[node] = 0

    #find max length node key
    maxstr = ''
    maxstrlength = 0
    for mstr in nodes.keys():
        if len(mstr) > maxstrlength:
            maxstr = mstr
            maxstrlength = len(mstr)

    #print maxstr, maxstrlength
    
    
    #construct current node
    mtree = tree.Tree()
    mtree.c = ' '.join(maxstr)
    
    mtree.depth = nodes[maxstr]
    del nodes[maxstr]

    #recurse with left and right children
    mtree.ch = getChildren(maxstr, nodes)

    #depth isn't fully annotated yet
    annotateDepth(mtree)
    return mtree

def annotateDepth(mtree):
  for child in mtree.ch:
    annotateDepth(child)
  if mtree.ch == []:
    mtree.depth = 0
  else:
    mtree.depth = max(ch.depth for ch in mtree.ch) + 1
  
def getChildren(maxkey, nodes):

    if len(maxkey) == 2 or len(nodes) == 0:
        return []
    #base case - nodes is empty
    #base case maxkey is length 2 / terminal
    else:
        #recurse case, find left and right child - add them and their children recursively

        #get cross of all possible child node combinations
        cross = []
        for nodea in nodes:
            for nodeb in nodes:
                if (nodea,nodeb) not in cross:
                    cross += [(nodea,nodeb)]

        #find left and right children s.t. union of set of 'words' == set of 'words' of maxstr
        lrchildren = ()
        for mchildren in cross:
            if set(mchildren[0]).union(set(mchildren[1])) == set(maxkey):
                lrchildren = mchildren
        assert lrchildren != ()

        #decide which child is left and which is right
        childA, childB = lrchildren
        #let left child be one with lowest "XX" number
        Aindices = [x for x in childA if x.isdigit()]
        Aindices.sort()
        Bindices = [x for x in childB if x.isdigit()]
        Bindices.sort()
        if Aindices[0] < Bindices[0]:
            lchildstr = childA
            rchildstr = childB
        else:
            lchildstr = childB
            rchildstr = childA

        #build current left and right, delete strs and get children recursively
        lchild = tree.Tree()
        lchild.c = ' '.join(lchildstr)
        lchild.depth = nodes[lchildstr]
        del nodes[lchildstr]
        lchild.ch = getChildren(lchildstr, nodes)

        rchild = tree.Tree()
        rchild.c = ' '.join(rchildstr)
        rchild.depth = nodes[rchildstr]
        del nodes[rchildstr]
        rchild.ch = getChildren(rchildstr, nodes)

        return [lchild, rchild]


def recurCompose(localVocab, relations, startingNode, steps = {}):
    curRelations = copy.deepcopy(relations)
    steps[localVocab[startingNode], startingNode] = 0
    curW = (localVocab[startingNode], startingNode)
    #print 'start with ', curW, steps[curW]
    for relation in relations:
        if relation != '00':
            for arg in relations[relation]:
                if arg[0] == startingNode:
                    try:
                        curRelations[relation].remove(arg)
                    except:
                        continue
                    prevW, steps, curRelations = recurCompose(localVocab, curRelations, startingNode = relation, steps = steps)
                    curW = prevW + curW
                    steps[curW] = steps[localVocab[startingNode], startingNode] = max(steps[prevW], steps[localVocab[startingNode], startingNode])+ 1
                    #print curW, steps[curW]

    if startingNode in relations:
        for relation in relations[startingNode]:
            try:
                curRelations[startingNode].remove(relation)
            except:
                continue
            prevW, steps, curRelations = recurCompose(localVocab, curRelations, startingNode = relation[0], steps = steps)
            curW = prevW + curW
            steps[curW] = steps[localVocab[startingNode], startingNode] = max(steps[prevW], steps[localVocab[startingNode], startingNode])+ 1
            #print curW, steps[curW]
    #print 'end with ', curW, steps[curW]
    return curW, steps, curRelations

if __name__ == '__main__':
    sentence = "00/1/03 01/0/N-lI:April 02/0/D-aN-lI:'s 02/1/03 02/2/01 03/0/N-aD-lI:flowers 04/0/V-aN-b{A-aN}-lI:are 04/=/03 05/0/D-lI:the 05/1/03 06/0/N-aD-lI:Sweet 06/=/07 07/0/N-aD-lI:Pea 08/0/X-cX-dX-lI:and 07/&/08 09/&/08 08/=/03 09/0/N-aD-lI:Daisy 10/0/.-lI:. 10/1/03"
    nodes = compose(sentence)
    mtree = nodes2Tree(nodes)
    #print mtree.c, mtree.ch, mtree.depth, mtree.vector
    print mtree.pprint()
