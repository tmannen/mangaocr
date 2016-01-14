def plot_img(regions, image):
	fig, ax = plt.subplots(ncols=1, nrows=1)
	ax.imshow(image, cmap=plt.cm.gray)

	for region in regions:

	    # skip small image

	    minr, minc, maxr, maxc = region.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
	                              fill=False, edgecolor='red', linewidth=1)
	    ax.add_patch(rect)

	plt.show()

def check_letters(label_image, regions, heightmap):
	letters = {}
	for region in regions:
		bbox = np.array(region.bbox)
		height = bbox[2] - bbox[0]
		width = bbox[3] - bbox[1]
		areas = [(-height, 0, -height, 0), (height, 0, height, 0),
				(0, width, 0, width), (0, -width, 0, -width)]

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
	firstletters = set()
	letter_labels = set(letters.keys())
	for key, region in letters.items():
		bbox = np.array(region.bbox)
		width = bbox[3] - bbox[1]
		height = bbox[2] - bbox[0]
		check_left = bbox + (0, -height, 0, -width)
		check_right = bbox + (0, width, 0, height)
		left_labels = set(np.unique(label_image[check_left[0]:check_left[2],check_left[1]:check_left[3]]))
		right_labels = set(np.unique(label_image[check_right[0]:check_right[2],check_right[1]:check_right[3]]))
		if left_labels.isdisjoint(letter_labels) and not right_labels.isdisjoint(letter_labels):
			firstletters.add(region)

	return firstletters

def get_regions(img):
	image = np.array(img)
	thresh = threshold_otsu(image)
	#bw = closing(image < thresh, square(3))
	bw = image < thresh
	label_image = label(bw, background=255)
	regions = [rgn for rgn in regionprops(label_image) if rgn.area > 15 
				and rgn.area < 500
				and get_std(rgn, image) > 85]
	heightmap = {rgn.label : rgn.bbox[2]-rgn.bbox[0] for rgn in regions}
	return [label_image, regions, heightmap]

#def check_overlap: kato ku addaat letterit, niin että teet 
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