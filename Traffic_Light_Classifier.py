import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import test_functions

# Image data directories
IMAGE_DIR_TRAINING = "/Users/almisfer/Documents/Traffic_Light_Classifier/traffic_light_images/training/"
IMAGE_DIR_TEST = "/Users/almisfer/Documents/Traffic_Light_Classifier/traffic_light_images/test/"

# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# Visualize the input images
image_shape = IMAGE_LIST[6][0]
image_label = IMAGE_LIST[6][1]
# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[6][0]
plt.imshow(selected_image)
print("The shape : " + str (image_shape.shape))
print("The label : " + str (image_label))


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32,32))

    return standard_im


## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
# Examples:
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):

    one_hot_encoded = []

    if label == 'red':
        one_hot_encoded = [1,0,0]
    elif label == 'yellow':
        one_hot_encoded = [0,1,0]
    elif label == 'green':
        one_hot_encoded = [0,0,1]

    return one_hot_encoded


def standardize(image_list):

    # Empty image data array
    standard_list = []

    # Iterate through all
    # the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## Display a standardized image and its label
image_index = 5
image_shape = STANDARDIZED_LIST[image_index][0]
image_label = STANDARDIZED_LIST[image_index][1]
selected_image = STANDARDIZED_LIST[image_index][0]
shape = IMAGE_LIST[image_index][0].shape
plt.imshow(selected_image)
print("The shape from STANDARDIZED_LIST : " + str (image_shape.shape))
print("The label from STANDARDIZED_LIST: " + str (image_label))
print("The shape from IMAGE_LIST:" + str (shape))
#print("The label from IMAGE_LIST:" + str ( IMAGE_LIST[image_index][0]) )


## Feature Extraction
image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert and image to HSV colorspace
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')


## Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):

    ## Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    ## Create and return a feature value and/or vector
    feature = []
    brightness = hsv[:,:,2]

    red_area = brightness[3:12, 11:-11]
    yellow_area = brightness[12:20, 11:-11]
    green_area = brightness[20:29, 11:-11]

    red_sum = np.sum(red_area)
    yellow_sum = np.sum(yellow_area)
    green_sum = np.sum(green_area)


    feature = [red_sum, yellow_sum, green_sum]

    return feature


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):

    ## Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = []

    feature = create_feature(rgb_image)
    feature_max = np.argmax(feature)

    if feature_max == 0:
        predicted_label = [1,0,0]
    elif feature_max == 1:
        predicted_label = [0,1,0]
    elif feature_max == 2:
        predicted_label = [0,0,1]

    return predicted_label


# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


## Determine the Accuracy


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# Visualize misclassified example(s)
num = 0
plt.imshow(MISCLASSIFIED[0][0])
print(MISCLASSIFIED[num][1])
print(MISCLASSIFIED[num][2])

feature1 = create_feature(MISCLASSIFIED[num][0])
feature2 = create_feature(MISCLASSIFIED[num][0])
feature3 = create_feature(MISCLASSIFIED[num][0])
feature = np.array(feature1) + np.array(feature2) +  np.array(feature3)
plt.imshow(MISCLASSIFIED[num][0])
print(feature1, feature2, feature3, feature)



if __name__ == "__main__":
    tests = test_functions.Tests()
    one_hot_encode =tests.test_one_hot(one_hot_encode)
    if(len(MISCLASSIFIED) > 0):
        result = tests.test_red_as_green(MISCLASSIFIED)
    else:
        print("MISCLASSIFIED may not have been populated with images.")