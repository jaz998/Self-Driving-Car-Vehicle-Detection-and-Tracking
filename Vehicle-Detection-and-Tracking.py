import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

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
# fig = plt.figure()
# plt.subplot(221)
# plt.imshow(img_car, cmap='gray')
# plt.title('Car Image')
# plt.subplot(222)
# plt.imshow(hog_image_car, cmap='gray')
# plt.title('HOG Visualization Car Image')
# plt.subplots_adjust(hspace=0.5)
# plt.subplot(223)
# plt.imshow(img_non_car, cmap='gray')
# plt.title('Non_car Image')
# plt.subplot(224)
# plt.imshow(hog_image_non_car, cmap='gray')
# plt.title('HOG Visualization non_car image')
# plt.show()

############ Combine and Normalize Features #########

# Define a function to compute binned color features
def bin_spatial(img, size=(64, 64)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(64, 64),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for item in imgs:
        # Read in each one by one
        img = mpimg.imread(item)
        # apply color conversion if other than 'RGB'
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(img, spatial_size)
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(img, hist_bins, hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


images_car = glob.glob('../vehicles/GTI_MiddleClose/*.png')
images_non_car = glob.glob('../non-vehicles/GTI/*.png')

cars = []
notcars = []
for image in images_car:
    # if 'image' in image or 'extra' in image:
    #     notcars.append(image)
    # else:
        cars.append(image)

for image in images_non_car:
    notcars.append(image)

car_features = extract_features(cars, cspace='RGB', spatial_size=(64, 64),
                                hist_bins=64, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(64, 64),
                                   hist_bins=64, hist_range=(0, 256))

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
else:
    print('Your function only returns empty feature vectors...')