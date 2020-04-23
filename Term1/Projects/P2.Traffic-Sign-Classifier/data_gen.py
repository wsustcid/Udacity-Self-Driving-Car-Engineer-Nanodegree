'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-21 10:28:20
@LastEditTime: 2020-04-22 21:54:45
'''
"""
[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(data_file, process=True):
    """ The pickled data is a dictionary with 4 key/value pairs:
      - features: is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
      - labels: is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
      - sizes: is a list containing tuples, (width, height) representing the original width and height the image.
      - coords: is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
    """
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)
        X, y = data['features'], data['labels']

    print("Data Loading...")
    print("{} data samples loaded from {}".format(X.shape[0], data_file))
    print("Image size is {}".format(X.shape[1:]))
    print("There are {} classes in this dataset".format(np.max(y)+1))
    print("Data Loading is done !")

    if process:
        X = data_processing(X, normalize=True, grayscale=True)
        print("The data is processed!")
        print("Now the shape of data is {}".format(X.shape))

    return X, y



def data_processing(X, normalize=True, grayscale=True):
    if grayscale:
        b = np.array([0.2989, 0.5870, 0.1140]) # RGB
        X = np.dot(X, b) # N,H,W
        ## add the channel axis
        X = np.expand_dims(X,axis=3)

    if normalize:
        X = (X-128.0)/128.0
    
    return X



def load_sign_names(sign_name_file):
    with open(sign_name_file, 'r') as f:
        read_lines = csv.reader(f)
        sign_name = []
        for line in read_lines:
            sign_name.append(line[1]) # the first line is column name!
    
    return sign_name[1:]



if __name__ == '__main__':
    
    ## Load data
    training_file = './traffic-signs-data/train.p'
    validation_file = './traffic-signs-data/valid.p'
    testing_file = './traffic-signs-data/test.p'
    sign_name_file = './traffic-signs-data/signnames.csv'

    X_train, y_train = load_data(training_file,process=False)
    X_valid, y_valid = load_data(validation_file,process=False)
    X_test,  y_test  = load_data(testing_file,process=False)
    sign_name = load_sign_names(sign_name_file)

    # visualize 36 images
    indexes = np.random.randint(0, X_train.shape[0], 8)
    visual_images = X_train[indexes]
    visual_labels = y_train[indexes]

    fig1, axes = plt.subplots(2,4)
    for i in range(2):
        for j in range(4):
            axes[i][j].imshow(visual_images[i*2+j])
            axes[i][j].set_title(sign_name[visual_labels[i*2+j]])
    
    ## label histogram
    fig2, axes2 = plt.subplots(1,3)
    axes2[0].hist(y_train, bins=43, rwidth=0.5)
    axes2[1].hist(y_valid, bins=43, rwidth=0.5)
    axes2[2].hist(y_test, bins=43, rwidth=0.5)
    axes2[0].set_title("Train labels distribution")
    axes2[1].set_title("Valid labels distribution")
    axes2[2].set_title("Test labels distribution")

    imgs_process = visual_images[:2]
    imgs_gray = data_processing(imgs_process, 
                                normalize=False, 
                                grayscale=True) 
    imgs_normal = data_processing(imgs_process, 
                                 normalize=True, 
                                 grayscale=True)
                                 
    fig3, axes3 = plt.subplots(2,3)
    axes3[0][0].imshow(imgs_process[0]) # rgb
    axes3[0][1].imshow(imgs_gray[0].reshape((32,32)), cmap='gray')
    axes3[0][2].imshow(imgs_normal[0].reshape((32,32)), cmap='gray')

    axes3[1][0].imshow(imgs_process[1])
    axes3[1][1].imshow(imgs_gray[1].reshape((32,32)), cmap='gray')
    axes3[1][2].imshow(imgs_normal[1].reshape((32,32)), cmap='gray')
    
    plt.show()

        

    