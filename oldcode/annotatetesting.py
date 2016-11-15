import json
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

gg = json.load(open("testannotations.json"))
img = Image.open("02-03.png").convert("L")
npimg = np.array(img)
annotations = gg[3]['annotations']
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.imshow(npimg, cmap=plt.cm.gray)
for a in annotations:
    ax.add_patch(mpatches.Rectangle((a['x'], a['y']), a['width'], a['height'], fill=False, linewidth=1))
plt.show()