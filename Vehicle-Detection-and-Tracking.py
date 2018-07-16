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
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        print('heatmap ', heatmap)
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        print('bbox[0]', bbox[0])
        print('bbox[1]', bbox[1])
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_boxes = True):
    coordinates_list = []

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        print('Scale = ', scale)
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))


    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)



    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)


            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)



            if test_prediction == 1 or show_boxes:
                randomColor = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), randomColor, 3)
                coordinates = ((xbox_left, (ytop_draw + ystart)), ((xbox_left + win_draw), (ytop_draw + win_draw + ystart)))
                coordinates_list.append(coordinates)
    #print('Printing the coordinates ', coordinates_list[0])

    return draw_img, coordinates_list

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    result = np.concatenate(img_features)
    return result

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
        return hog_features



# Define a function to extract features from a list of images
def extract_features(imgs, cspace ='RGB', orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0, spatial_size=(64, 64), hist_bins=32, hist_range=(0, 256)):

    # Create a list to append feature vector to
    features = []
    # Iterate through the list of images
    count = 0
    for file in imgs:
        file_features = []
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

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Color Features
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, spatial_size)
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        # Append the new feature vector to the features list
        #features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        file_features.append(spatial_features)
        file_features.append(hist_features)
        file_features.append(hog_features)


        features.append(np.concatenate(file_features))
        count = count + 1
        print('image ', count)
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

# Mac OS Path
images_car = glob.glob('../vehicles/GTI_MiddleClose/*.png')
images_non_car = glob.glob('../non-vehicles/GTI/*.png')
images = glob.glob(('../all_images/*vehicles/*/*'))

# Windows Path
# images_car = glob.glob('..\\vehicles\\GTI_MiddleClose\\*.png')
# images_non_car = glob.glob('..\\non-vehicles\\GTI\\*.png')
# images = glob.glob(('..\\all_images\\*vehicles\\*\\*'))

cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)




# Use a smaller sample for testing hog features classifier
sample_size = 5
hog_cars = cars[0:sample_size]
hog_notcars = notcars[0:sample_size]

# Full sample
# hog_cars = cars
# hog_notcars = notcars

print('Number of car_images', len(hog_cars))
print('Number of non_car images:', len(hog_notcars))

# print('***********Showing test image**************')
# test_img = mpimg.imread(hog_cars[0])
# plt.title('Loading image for classifier')
# plt.imshow(test_img)
# plt.show()



# Defining the parameters for hog features
# colorspace = 'YCrCb'
# orient = 9
# pix_per_cell = 8
# cell_per_block = 2
# hog_channel = 'ALL'
# # Specifying spatial and histbin parameters
# spatial = (32, 32)
# histbin = 32

# Alternative parameters values
# colorspace = 'YCrCb'
# orient = 9
# pix_per_cell = 16
# cell_per_block = 2
# hog_channel = 'ALL'
# # Specifying spatial and histbin parameters
# spatial = (32, 32)
# histbin = 32

# Trying Parameter values
colorspace = 'YCrCb'
orient = 10
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
# Specifying spatial and histbin parameters
spatial = (16, 16)
histbin = 16




# Extracting hog features
t = time.time()
hog_car_features = extract_features(hog_cars, cspace=colorspace, orient = orient, pix_per_cell= pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial, hist_bins=histbin)
hog_notcar_features = extract_features(hog_notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial, hist_bins=histbin)
t2=time.time()
print(round(t2-t),2), ' seconds to extract hog features'






# color_car_features = extract_features_color(hog_cars, cspace='RGB', spatial_size=(spatial, spatial),
#                                       hist_bins=histbin, hist_range=(0, 256))
# color_notcar_features = extract_features_color(hog_notcars, cspace='RGB', spatial_size=(spatial, spatial),
#                                          hist_bins=histbin, hist_range=(0, 256))

# print('hog car features shape', hog_car_features[0].shape)
# print('color car features shape', color_car_features[0].shape)
# combined_notcar_features = np.concatenate((hog_notcar_features, color_notcar_features))




########### training a color classifier #################

# Create an array stack of feature vectors
X = np.vstack((hog_car_features, hog_notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(hog_car_features)), np.zeros(len(hog_notcar_features))))

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
print('Features vector length:', len(X_train[0]))
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
n_predict = 1000
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

######### Search and Classify #####################

#print('hog notcar features', hog_notcar_features[0].shape)
# print('not car feature len', len(notcar_features.shape()))
image1 = mpimg.imread('../test_images/test1.jpg') # Mac OS path
image2 = mpimg.imread('../test_images/test2.jpg')
image3 = mpimg.imread('../test_images/test3.jpg')
image4 = mpimg.imread('../test_images/test4.jpg')
image5 = mpimg.imread('../test_images/test5.jpg')
image6 = mpimg.imread('../test_images/test6.jpg')
#image2 = mpimg.imread('..\\test_images\\test1.jpg') # Windows Path
# draw_image = np.copy(image2)




# windows = slide_window(image2, x_start_stop=[None, None], y_start_stop=[350, 720],
#                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#
# hot_windows = search_windows(image2, windows, svc, X_scaler, color_space=colorspace,
#                         spatial_size= spatial, hist_bins=histbin,
#                         orient=orient, pix_per_cell=pix_per_cell,
#                         cell_per_block=cell_per_block,
#                         hog_channel=hog_channel, spatial_feat=True,
#                         hist_feat=True, hog_feat=True)
#
# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

# print('Showing sliding windows image')
# plt.imshow(window_img)
# plt.show()


########################### Hog Sub-sampling Window Search ###################################

# Defining variable values

# scale = 1.5

# out_img, coordinates_list = find_cars(image2, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial, histbin)
# # print('Hog Sub-sampling Window Search image')
# # plt.imshow(out_img)
# # plt.show()
#
# heat = np.zeros_like(image2[:,:,0]).astype(np.float)
#
# # Add heat to each box in box list
# heat = add_heat(heat, coordinates_list)
#
# # Apply threshold to help remove false positives
# heat = apply_threshold(heat, 1)
#
# # Visualize the heatmap when displaying
# heatmap = np.clip(heat, 0, 255)
#
# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(image2), labels)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()
# plt.show()

def process_frame(frame):

    coordinates_list_combo = []

    # Parameters combination 1
    ystart = 400
    ystop = 500
    scale =1.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)



    # Parameters combination 2
    ystart = 430
    ystop = 530
    scale =1.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)



    # Parameters combination 3
    ystart = 460
    ystop = 560
    scale =1.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)



    # Parameters combination 4
    ystart = 400
    ystop = 480
    scale =1

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)


    # Parameters combination 5
    ystart = 430
    ystop = 510
    scale =1

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)



    # Parameters combination 6
    ystart = 370
    ystop = 540
    scale =2.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)


    # Parameters combination 7
    ystart = 400
    ystop = 570
    scale =2.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)



    # Parameters combination 8
    ystart = 430
    ystop = 650
    scale =2.5

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)


    # Parameters combination 9
    ystart = 450
    ystop = 650
    scale = 3

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)

    # Parameters combination 10
    ystart = 400
    ystop = 500
    scale =1.2

    out_img, coordinates_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial, histbin)

    coordinates_list_combo.append(coordinates_list)

    # #Display the image as an example
    # plt.title('Parameters Image')
    # plt.imshow(out_img)
    # plt.show()

    boxes =  [item for sublist in coordinates_list_combo for item in sublist] # flatten a list of list



    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)


    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    # plt.imshow(draw_img)
    # plt.show()
    return draw_img

processed_image1 = process_frame(image1)
print('showing processed image 1')
# plt.title('test image 1')
# plt.imshow(processed_image1)
# plt.show()
#
processed_image2 = process_frame(image2)
print('showing processed image 2')
# plt.title('test image 2')
# plt.imshow(processed_image2)
# plt.show()
#
processed_image3 = process_frame(image3)
print('showing processed image 3')
# plt.title('test image 3')
# plt.imshow(processed_image3)
# plt.show()
#
processed_image4 = process_frame(image4)
print('showing processed image 4')
# plt.title('test image 4')
# plt.imshow(processed_image4)
# plt.show()
#
processed_image5 = process_frame(image5)
print('showing processed image 5')
# plt.title('test image 5')
# plt.imshow(processed_image5)
# plt.show()
#
processed_image6 = process_frame(image6)
print('showing processed image 6')
# plt.title('test image 6')
# plt.imshow(processed_image6)
# plt.show()



####################### Multiple Windows Search using the Find Cars Function ###########################################

# test_img_multi_win = mpimg.imread('..\\test_images\\test1.jpg') # Windows path
# test_img_multi_win = mpimg.imread('../test_images/test1.jpg') # Mac OS path
# draw_img, out_img = process_frame(test_img_multi_win)
# plt.imshow(draw_img)
# plt.show()

# print('Showing out img')
# plt.title('Out img')
# plt.imshow(out_img)
# plt.show()
# print('Showing draw img')
# plt.title('draw img')
# plt.imshow(draw_img)
# plt.show()
# cv2.waitKey()



# ############ Read the video #################################

# Mac OS Path
project_video = '../project_video.mp4'
output_video = '../test_videos_output/output_video_v12.mp4'

# Windows Path
# project_video = '..\\project_video.mp4'
# output_video = '..\\test_videos_output\\output_video_v8.mp4'
# #clip1 = VideoFileClip(project_video).subclip(0,3)
# #clip1 = VideoFileClip(project_video).subclip(38, 42)
#clip1 = VideoFileClip(project_video)
# # # #print("###################Now running processing frame - video#######")
#processed_clip1 = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
#processed_clip1.write_videofile(output_video, audio=False)

# result_process_frame = process_frame(road_image)
# cv2.imshow("Result processed frame", result_process_frame)
# cv2.waitKey()









