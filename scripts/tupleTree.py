class Tree(object):
  def __init__(self, p=None, ch=None):
    self.parent = p
    self.children = ch
    if ch:
      self.valence = len(ch)
    else:
      self.valence = 0
    self.data = None
    self.depth = None #depth is counted from the leaves, so it's a measure of abstractness
    self.head = None

  def sprout(self, numkids):
    #grow [numkids] child tuple nodes
    self.children = (Tree(p=self),)*numkids
    self.valence = numkids

  def getDepth(self):
    #return the depth of the current node
    if not self.depth:
      self.depth = max(ch.getDepth() for ch in self.children) + 1
    return(self.depth)

  def setHead(self, head):
    #tells the node the index of the child that heads it
    self.Head = head
