import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

# Read in our vehicles
#car_images = glob.glob('*.jpeg')
image_path＿car = './images/car_example.png'
img_car = cv2.imread(image_path＿car)
img_car = np.copy(img_car)
image_path_non_car = './images/non_car_example.png'
img_non_car = np.copy(cv2.imread(image_path_non_car))


# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize= True
# Otherwise there is no second return element

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """

    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=False,
                      visualise=vis, feature_vector=feature_vec)

    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features



gray_car = cv2.cvtColor(img_car, cv2.COLOR_BGR2GRAY)
gray_non_car = cv2.cvtColor(img_non_car, cv2.COLOR_BGR2GRAY)

features_car, hog_image_car = get_hog_features(gray_car, orient=9,
                                       pix_per_cell=8, cell_per_block=2,
                                       vis=True, feature_vec=False)

features_non_car, hog_image_non_car = get_hog_features(gray_non_car, orient=9,
                                       pix_per_cell=8, cell_per_block=2,
                                       vis=True, feature_vec=False)

# Plot the examples
fig = plt.figure()
plt.subplot(221)
plt.imshow(img_car, cmap='gray')
plt.title('Car Image')
plt.subplot(222)
plt.imshow(hog_image_car, cmap='gray')
plt.title('HOG Visualization Car Image')
plt.subplots_adjust(hspace=0.5)
plt.subplot(223)
plt.imshow(img_non_car, cmap='gray')
plt.title('Non_car Image')
plt.subplot(224)
plt.imshow(hog_image_non_car, cmap='gray')
plt.title('HOG Visualization non_car image')
plt.show()