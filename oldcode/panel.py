from testing import *

img = Image.open("../test.png").convert("L")
image = np.array(img)

values = get_regions(image)

letters = check_letters(values['labeled_image'], values['regions'], values['heightmap'])
firstletters = first_letters(values['labeled_image'], letters)
new_label_image, merged = merge_regions(values['labeled_image'], firstletters, letters)
plot_img(merged, image)

#from skimage.transform import rotate
#plt.imshow(rotate(get_area(merged[5], image), 90, resize=True), cmap=plt.cm.gray)
#plt.imshow(image, cmap=plt.cm.gray)
#plt.show()
