class Tree(object):
  def __init__(self, ch=None):
    self.children = ch
    if ch:
      self.valence = len(ch)
    else:
      self.valence = 0
    self.data = None

  def sprout(self, numkids):
    #grow [numkids] child tuple nodes
    self.children = (Tree(),)*numkids
    self.valence = numkids
