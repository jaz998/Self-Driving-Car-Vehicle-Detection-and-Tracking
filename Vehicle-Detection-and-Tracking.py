import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

# Read in our vehicles
#car_images = glob.glob('*.jpeg')
image_path = './images/car_example.png'
#image = mpimg.imread(image_path)
image = cv2.imread(image_path)
print("Image shape ", image.shape)
new_img = cv2.resize(image, (320, 320))


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



# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)


# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(new_img, orient=9,
                                       pix_per_cell=8, cell_per_block=2,
                                       vis=True, feature_vec=False)

cv2.imwrite('./images/hog_car_example.png', hog_image)

cv2.imshow('new_image', new_img)
cv2.waitKey()
cv2.imshow('hog_image', hog_image)
cv2.waitKey()