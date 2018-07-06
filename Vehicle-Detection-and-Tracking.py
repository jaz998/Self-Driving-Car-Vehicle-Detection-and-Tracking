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
# image_path＿car = './images/car_example.png'
# img_car = cv2.imread(image_path＿car)
# img_car = np.copy(img_car)
# image_path_non_car = './images/non_car_example.png'
# img_non_car = np.copy(cv2.imread(image_path_non_car))


# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize= True
# Otherwise there is no second return element

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=True):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """

    hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=True,
                      visualise=True, feature_vector=feature_vec)



    if vis:
        return hog_features, hog_image
    else:
        print('Get hog features', hog_features)
        return hog_features



# Define a function to extract features from a list of images
def extract_hog_features(imgs, cspace = 'RGB', orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0):
    # Create a list to append feature vector to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        #image = cv2.imread(file)
        # apply color conversion if it's not 'rgb'
        if cspace!='RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
            #feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('Feature image shape', feature_image.shape)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            print('feature image channel 0', feature_image[:,:,hog_channel].shape)
        features.append(hog_features)
        print('features', features)
    return features

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
def extract_features_color(imgs, cspace='RGB', spatial_size=(64, 64),
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




# Use a smaller sample for testing hog features classifier
sample_size = 500
hog_cars = cars[0:sample_size]
hog_notcars = notcars[0:sample_size]

print('Number of car_images', len(hog_cars))
print('Number of non_car images:', len(hog_notcars))
print('hog car[0]', cars[0])

# print('hog cars shape', hog_cars[0].shape)
# print('hog notcars shape', hog_notcars[0].shape)


# Defining the parameters for hog features
colorspace = 'RGB'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0


# Extracting hog features
t = time.time()
hog_car_features = extract_hog_features(hog_cars, cspace=colorspace, orient = orient, pix_per_cell= pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
hog_notcar_features = extract_hog_features(hog_notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
t2=time.time()
print(round(t2-t),2), ' seconds to extract hog features'




# Specifying spatial and histbin parameters
spatial = 32
histbin = 32

color_car_features = extract_features_color(cars, cspace='RGB', spatial_size=(spatial, spatial),
                                      hist_bins=histbin, hist_range=(0, 256))
color_notcar_features = extract_features_color(notcars, cspace='RGB', spatial_size=(spatial, spatial),
                                         hist_bins=histbin, hist_range=(0, 256))


print('hog car features', hog_car_features[0])

combined_car_features = np.concatenate(hog_car_features, color_car_features)
combined_notcar_features = np.concatenate(hog_notcar_features, color_notcar_features)


#print('hog notcar features', hog_notcar_features[0].shape)
# print('not car feature len', len(notcar_features.shape()))

########### training a color classifier #################

# Create an array stack of feature vectors
X = np.vstack((combined_car_features, combined_notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(combined_car_features)), np.zeros(len(combined_notcar_features))))

print('X shape', X.shape)
print ('y shape', y.shape)

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

######### Sliding Windows Search #####################






