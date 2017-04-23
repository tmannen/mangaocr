import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

from PIL import Image
import numpy as np
import os

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

def get_area(region, image):
	top, left, bottom, right = region.bbox
	return image[top:bottom,left:right]

def plot_img(regions, image):
	fig, ax = plt.subplots(ncols=1, nrows=1)
	ax.imshow(image, cmap=plt.cm.gray)

	for region in regions:
	    minr, minc, maxr, maxc = region.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
	                              fill=False, edgecolor='red', linewidth=1)
	    ax.add_patch(rect)

	plt.show()

def get_ccs(image):
	thresh = threshold_otsu(image)
	bw = image < thresh
	label_image = label(bw)
	props = regionprops(label_image)
	for prop in props: prop.orig_image = image
	return props


def check_letters(label_image, regions, heightmap):
	letters = {}
	for region in regions:
		bbox = np.array(region.bbox)
		height = bbox[2] - bbox[0]
		width = bbox[3] - bbox[1]
		#check only left and right for now
		areas = [(0, width, 0, width), (0, -width, 0, -width)]

		for area in areas:
			check = bbox + area
			to_check = label_image[check[0]:check[2],check[1]:check[3]]
			labels = np.unique(to_check)
			for label in labels:
				try:
					if label != 0 and abs(heightmap[label]-height) < 0.7*height:
						letters[region.label] = region
				except KeyError:
					pass

	return letters

def first_letters(label_image, letters):
	firstletters = list()
	letter_labels = set(letters.keys())
	for key, region in letters.items():
		bbox = np.array(region.bbox)
		width = bbox[3] - bbox[1]
		height = bbox[2] - bbox[0]
		check_left = bbox - (0, height, 0, width)
		check_right = bbox + (0, width, 0, height)
		left_labels = set(np.unique(label_image[check_left[0]:check_left[2],check_left[1]:check_left[3]]))
		right_labels = set(np.unique(label_image[check_right[0]:check_right[2],check_right[1]:check_right[3]]))
		if left_labels.isdisjoint(letter_labels) and not right_labels.isdisjoint(letter_labels):
			firstletters.append(region)

	return firstletters

def get_regions(image):
	thresh = threshold_otsu(image)
	x, y = image.shape
	maxheight = float(y/15)
	#grayscale image, so 255=white, 0=black. binarize for labeling, 0 (False) considered background:
	bw = image < thresh
	label_image = label(bw)
	regions = [rgn for rgn in regionprops(label_image)]

	heightmap = {rgn.label : rgn.bbox[2]-rgn.bbox[0] for rgn in regions}
	return {'labeled_image': label_image,
		    'regions': regions,
		    'heightmap': heightmap
		    }

#def check_overlap: kato ku addaat letterit, niin etta teet 
#label imagen jossa lasket kuina moneen osuu bbox ja sitten vertaat
#centroid

def merge_regions(label_image, firstletters, letters):
	new_label_image = np.zeros_like(label_image)
	for letter in firstletters:
		regions = set()
		bbox = np.array(letter.bbox)
		width = bbox[2] - bbox[0]
		check_right = bbox + (0, width, 0, width)
		overlaps = np.unique(label_image[check_right[0]:check_right[2],check_right[1]:check_right[3]])
		
		while True:
			intersection = [x for x in overlaps if x in letters
							and vertical_check(letter, letters[x])]
			if len(intersection) <= 0:
				break

			regions.add(letter.label)
			for label in intersection: regions.add(label)
			check_right += (0, width, 0, width)
			overlaps = np.unique(label_image[check_right[0]:check_right[2],check_right[1]:check_right[3]])

		for label in regions:
			new_label_image[label_image==label] = letter.label

	return new_label_image, regionprops(new_label_image)

#yhdista saadut linet
def clump_lines(label_image, lines):
	new_label_image = np.zeros_like(label_image)
	for line in lines:
		regions = set()
		bbox = np.array(line.bbox)
		height = bbox[3] - bbox[1] + 5
		check_down = bbox + (height, 0, height, 0)
		overlaps = np.unique(label_image[check_down[0]:check_down[2],check_down[1]:check_down[3]])
		
		while True:
			intersection = [x for x in overlaps if x!=0]
			if len(intersection) <= 0:
				break

			regions.add(line.label)
			for label in intersection: regions.add(label)
			check_down += (height, 0, height, 0)
			overlaps = np.unique(label_image[check_down[0]:check_down[2],check_down[1]:check_down[3]])

		for label in regions:
			new_label_image[label_image==label] = line.label

	return regionprops(new_label_image)

def vertical_check(region1, region2):
	minr, minc, maxr, maxc = region1.bbox
	return minr < region2.centroid[0] < maxr

def crop_area(pilimg, bbox, pixels):
	return pilimg.crop((bbox[1]-pixels, bbox[0]-pixels, bbox[3]+pixels, bbox[2]+pixels))

def save_text(image, regions):
	pilimg = Image.fromarray(image)
	count = 0
	for region in regions:
		crop_area(pilimg, region.bbox, 1).save("texts/" + str(count) + ".jpg", "JPEG")
		count += 1

def get_texts(image, regions):
	pilimg = Image.fromarray(image)
	return [crop_area(pilimg, region.bbox, 1) for region in regions]

def edge_density_variation(region, image):
	pass

def get_parameters(region, image):
	width = region.bbox[2] - region.bbox[0]
	height = region.bbox[3] - region.bbox[1]
	aspect_ratio = width/float(height)
	std = np.std(get_area(region, image))
	carea_area_ratio = region.convex_area/region.area
	ecc = region.eccentricity
	ori = region.orientation
	soli = region.solidity
	moments_hu = [mom for mom in region.moments_hu.flatten()]
	paralist = moments_hu+[carea_area_ratio, ori, soli, aspect_ratio, std, ecc]
	return np.array(paralist).reshape(1, -1)

def classify(region, image, classifier):
	pars = get_parameters(region, image)
	probs = classifier.predict_proba(pars)
	neg, pos = probs[0, 0], probs[0, 1]
	if pos > 0.2: return 1
	else: return 0

def save_parameters(regions, filename, positive):
	if not os.path.isfile(filename):
		with open(filename, "w") as datafile:
			region = regions[0]
			moments = region.moments
			moments_str = ["moment" + str(ind) for ind, x in enumerate(moments.flatten())]
			moments_n = region.moments_normalized
			moments_n_str = ["moment_norm" + str(ind) for ind, x in enumerate(moments_n.flatten())]
			moments_hu = region.moments_hu
			moments_hu_str = ["moment_hu" + str(ind) for ind, x in enumerate(moments_hu.flatten())]
			csvstring = ",".join(["width", "height", "aspect_r", "std", "area", "convex_area",
				"eccentricity", "equi_diameter", "euler_n", "extent", "filled_area",
				"major_axis_length", "minor_axis_length", "orientation", "perimeter", "solidity"]
				+moments_str+moments_n_str+moments_hu_str+["y"]
				)
			datafile.write(csvstring+"\n")

	with open(filename, "a") as datafile:
		for region in regions:
			width = region.bbox[2] - region.bbox[0]
			height = region.bbox[3] - region.bbox[1]
			aspect_ratio = width/float(height)
			std = np.std(get_area(region, region.orig_image))
			area = region.area
			carea = region.convex_area
			ecc = region.eccentricity
			eq_dia = region.equivalent_diameter
			en = region.euler_number
			extent = region.extent
			farea = region.filled_area
			mal = region.major_axis_length
			mil = region.minor_axis_length
			ori = region.orientation
			peri = region.perimeter
			soli = region.solidity
			moments = [str(mom) for mom in region.moments.flatten()]
			moments_n = [str(mom) for mom in region.moments_normalized.flatten()]
			moments_hu = [str(mom) for mom in region.moments_hu.flatten()]
			y = 1 if positive else 0
			paralist = [width, height, aspect_ratio, std, area, carea, ecc, eq_dia,en,extent
			,farea,mal,mil,ori,peri,soli]+moments+moments_n+moments_hu+[str(y)]
			csvrow = ",".join([str(x) for x in paralist])
			datafile.write(csvrow+"\n")
