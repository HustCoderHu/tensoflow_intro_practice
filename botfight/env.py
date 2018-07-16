import numpy as np

class Env():
  def __init__(sefl):
    self.x = 1
    self.y = 1
    self.map = np.zeros([12, 12], np.int8)
    self.info = 111
    # end init

  def step(self, x, y):
    return observation, reward, done, info
    # end step

  def render(self):
    # end render

  # end


