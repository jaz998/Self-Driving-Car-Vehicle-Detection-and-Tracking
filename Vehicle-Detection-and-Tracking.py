import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split




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


########## Loading the data ###########

images_car = glob.glob('../vehicles/GTI_MiddleClose/*.png')
images_non_car = glob.glob('../non-vehicles/GTI/*.png')
images = glob.glob(('../all_images/*vehicles/*/*'))

cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)

print('Number of non_car images:', len(notcars))
print('Number of car_images', len(cars))

# Specifying spatial and histbin parameters
spatial = 32
histbin = 32

car_features = extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial),
                                hist_bins=histbin, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial),
                                   hist_bins=histbin, hist_range=(0, 256))

# if len(car_features) > 0:
#     # Create an array stack of feature vectors
#     X = np.vstack((car_features, notcar_features)).astype(np.float64)
#     # Fit a per-column scaler
#     X_scaler = StandardScaler().fit(X)
#     # Apply the scaler to X
#     scaled_X = X_scaler.transform(X)
#     car_ind = np.random.randint(0, len(cars))
#     # Plot an example of raw and scaled features
#     fig = plt.figure(figsize=(12, 4))
#     plt.subplot(131)
#     plt.imshow(mpimg.imread(cars[car_ind]))
#     plt.title('Original Image')
#     plt.subplot(132)
#     plt.plot(X[car_ind])
#     plt.title('Raw Features')
#     plt.subplot(133)
#     plt.plot(scaled_X[car_ind])
#     plt.title('Normalized Features')
#     fig.tight_layout()
#     plt.show()
# else:
#     print('Your function only returns empty feature vectors...')

########### training a color classifier #################

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0,100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=rand_state)

# Fit a per-column scaler only on the training data
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using spatial binning of:', spatial, ' and ', histbin, ' histogram bins')
print('Features vecotr length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test accuracy of SVC =', round(svc.score(X_test, y_test), 4))
# Check the prediction of a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

