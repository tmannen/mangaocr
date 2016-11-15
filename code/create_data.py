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
	masks = {}
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
		
		masks[fname] = mask

	return masks

def get_positive_training_imgs(fname, mask, dirpath = "data/images/"):
	"""
	This function should return random (?) 32x32 or some other size patches from the images.
	For positive examples I was thinking of choosing 1s from the mask randomly and then checking
	if any of the 32x32 pixels are 0. if they are, dont crop and try again. same with 0s for negative

	With replacement or not?
	"""

	ones = np.where(mask)
	max_x, max_y = mask.shape
	img = np.array(Image.open(dirpath + fname).convert("L"))
	imgs = []
	picks = set()
	for i in range(100):
		pick = np.random.randint(len(ones[0]))
		if pick in picks:
			continue
		picks.add(pick)
		x = ones[0][pick]
		y = ones[1][pick]
		candidate = mask[x:x+32, y:y+32] if x+32 < max_x and y+32 < max_y else 0
		if np.all(candidate):
			imgs.append(img[x:x+32, y:y+32])

	return imgs

def get_negative_training_imgs(fname, mask, dirpath = "data/images/"):
	zeros = np.where(mask==0)
	max_x, max_y = mask.shape
	img = np.array(Image.open(dirpath + fname).convert("L"))
	imgs = []
	picks = set()

	for i in range(100):
		pick = np.random.randint(len(zeros[0]))
		if pick in picks:
			continue
		picks.add(pick)
		x = zeros[0][pick]
		y = zeros[1][pick]
		candidate = mask[x:x+32, y:y+32] if x+32 < max_x and y+32 < max_y else 1
		if not np.any(candidate):
			imgs.append(img[x:x+32, y:y+32])

	return imgs

def get_training():
	masks1 = create_masks("0-13annotations.json")
	masks2 = create_masks("14-41annotations.json")

	masks = masks1 + masks2
	#TODO: with replacement? kerää näitä ja kokeile keras mallia