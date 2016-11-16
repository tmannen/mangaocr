import json
import numpy as np
from PIL import Image

def cut_images(annotation_file, dirpath="data/images/"):
	data = json.load(open(dirpath + annotation_file))
	files = [x for x in data if len(x['annotations']) > 0] #annotations for each file
	crops = []
	for a in files: #go through each file
		img = Image.open(dirpath + a['filename']).convert("L")
		for ann in a['annotations']:
			cropbox = (ann['x'], ann['y'], ann['x'] + ann['width'], ann['y'] + ann['height'])
			crops.append(img.crop(cropbox))

	return crops

def create_masks(annotation_file, dirpath="data/images/"):
	data = json.load(open(dirpath + annotation_file))
	files = [x for x in data if len(x['annotations']) > 0] #annotations for each file
	masks = []
	for a in files: #go through each file
		fname = a['filename']
		img = Image.open(dirpath + a['filename']).convert("L")
		mask = np.zeros_like(np.array(img))
		for ann in a['annotations']:
			ux = int(ann['x']) #upper left x coordinate, and so on
			uy = int(ann['y'])
			lx = int(ann['x'] + ann['width'])
			ly = int(ann['y'] + ann['height'])
			mask[uy:ly, ux:lx] = 1
		
		masks.append((fname, mask))

	return masks

def get_training_imgs(fname, mask, amount, positive, dirpath = "data/images/"):
	"""
	This function should return random (?) 32x32 or some other size patches from the images.
	For positive examples I was thinking of choosing 1s from the mask randomly and then checking
	if any of the 32x32 pixels are 0. if they are, dont crop and try again. same with 0s for negative
	"""

	ones = np.where(mask)
	zeros = np.where(mask==0)
	max_x, max_y = mask.shape
	img = np.array(Image.open(dirpath + fname).convert("L"))
	imgs = []
	picks = set()
	itercount = 0
	print("getting some images")
	#take 100 images or if not possible stop at 500 iterations
	while len(imgs) < amount and itercount < 500:
		pick = np.random.randint(len(ones[0]))
		if pick in picks:
			continue
		picks.add(pick)

		if positive:
			x = ones[0][pick]
			y = ones[1][pick]
			candidate = mask[x:x+32, y:y+32] if x+32 < max_x and y+32 < max_y else 0
			if np.all(candidate):
				imgs.append((img[x:x+32, y:y+32], 1))
		else:
			x = zeros[0][pick]
			y = zeros[1][pick]
			candidate = mask[x:x+32, y:y+32] if x+32 < max_x and y+32 < max_y else 1
			if not np.any(candidate):
				imgs.append((img[x:x+32, y:y+32], 0))

		itercount += 1

	return imgs

def get_training():
	masks1 = create_masks("0-13annotations.json")
	masks2 = create_masks("14-41annotations.json")
	images = []
	masks = masks1 + masks2
	for m in masks:
		fname, mask = m

		#get 100 positive and false examples from each image. (32x32 cuts)
		print("processing image %s", fname)
		images += get_training_imgs(fname, mask, amount=100, positive=True)
		images += get_training_imgs(fname, mask, amount=100, positive=False)

	return images

training = get_training()