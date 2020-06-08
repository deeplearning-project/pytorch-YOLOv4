import torchvision.transforms as T
import numpy as np

class RandomResize(object):
    """
    Resize image to an random size.
    """
    def __init__(self, short_size=[256, 480]):
        self.bot_size = short_size[0]
        self.top_size = short_size[1]
    
    def __call__(self, image):
        rand_size = np.random.randint(self.bot_size, high=self.top_size)
        width, height = image.size
        if width < height:
            height = int(rand_size/width * height)
            width = rand_size
        else:
            width = int(rand_size/height * width)
            height = rand_size
        
        return T.Resize((height, width))(image)

        
