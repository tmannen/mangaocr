##TODO: tsekkaa thresholding että black ja white, pitikö olla pienempi vai suurempi?

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

from PIL import Image
import numpy as np

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import io, color

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square

from testing import *

img = Image.open("data/Onepiece/05.png").convert("L")