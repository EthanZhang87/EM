from mnist import Mnist
import numpy as np


i,j = 0,1

mnist = Mnist()
images, labels = mnist.image_labels_of_digits([0,1])

mnist.plot_image(images[0])