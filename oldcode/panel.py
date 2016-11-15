from testing import *

img = Image.open("data/fuwa.jpg").convert("L")
image = np.array(img)

values = get_regions(image)
letters = check_letters(values[0], values[1], values[2])
firstletters = first_letters(values[0], letters)
new_label_image, merged = merge_regions(values[0], firstletters, letters)

#from skimage.transform import rotate
#plt.imshow(rotate(get_area(merged[5], image), 90, resize=True), cmap=plt.cm.gray)
#plt.imshow(image, cmap=plt.cm.gray)
#plt.show()
