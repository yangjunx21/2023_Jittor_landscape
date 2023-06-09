from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# example 3386875748_27e86864ba_b.png

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--picture', help="name of the picture", type=str)
args = parser.parse_args()

img = Image.open('../data/train_resized/labels/'+args.picture)
img = np.array(img)
if img.ndim == 3:
    img = img[:,:,0]
plt.imshow(img)
#plt.show()
plt.savefig("test.png")