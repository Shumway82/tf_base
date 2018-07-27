import random
import copy
import numpy as np

class ImagePool_(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image