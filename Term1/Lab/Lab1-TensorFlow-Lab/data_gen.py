'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-04-19 23:58:52
@LastEditTime: 2020-04-20 11:45:01
'''

"""
Goal:
  1. Download and unzip the dataset, and saved them to the pickle file.  
  2. Next time we no longer have to start from the beginning.  
     Just run the code below and it will load all the data.

Hint:
  1. MD5文件校验的作用：每个文件都可以用MD5验证程序算出一个固定的MD5值，是独一无二的。 一般来说，开发方会在软件发布时预先算出文件的MD5值，如果文件被盗用，加了木马或者被篡改版权，那么它的MD5值也随之改变，也就是说我们对比文件当前的MD5值和它标准的MD5值来检验它是否正确和完整.
  2. Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
"""


import os

from urllib.request import urlretrieve
import hashlib

from zipfile import ZipFile
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image

from sklearn.preprocessing import LabelBinarizer

import pickle



def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

    # Wait until you see that all files have been downloaded.
    print('All files downloaded.')


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
        
        # Wait until you see that all features and labels have been uncompressed.
        print('All features and labels uncompressed.')
                
    return np.array(features), np.array(labels)



def normalize_grayscale(image_data):
    """
    Normalize the image data with "Min-Max scaling" to a range of [a=0.1, b=0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    a = 0.1
    b = 0.9
    x_min = 0
    x_max = 255 # can not use np.max, may be one image do not has this value
    
    return a + (image_data-x_min)*(b-a)/(x_max-x_min)


if __name__ == '__main__':
    
    train_data_file = './dataset/notMNIST_train.zip'
    test_data_file  = './dataset/notMNIST_test.zip' 

    train_data_md5 = 'c8673b3f28f489e9cdf3a3d74e2ac8fa'
    test_data_md5 = '5d3c7e653e63471c88df796156a9dfa9' 
    
    ## Download the training and test dataset.
    download(url='https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', file=train_data_file)
    download(url='https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', file=test_data_file)
    
    ## Verify the md5
    # Make sure the files aren't corrupted, use comma to add Error output
    assert hashlib.md5(open(train_data_file, 'rb').read()).hexdigest() == train_data_md5, 'notMNIST_train.zip file is corrupted.'
    assert hashlib.md5(open(test_data_file, 'rb').read()).hexdigest() == test_data_md5, 'notMNIST_test.zip file is corrupted.'


    ## Get the features and labels from the zip files
    train_features, train_labels = uncompress_features_labels(train_data_file)
    test_features, test_labels = uncompress_features_labels(test_data_file)

    ## Limit the amount of data to work with a docker container
    docker_size_limit = 150000
    train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)


    ## Implement Min-Max scaling for grayscale image data
    # Set flags for feature engineering. prevent you from skipping an important step.
    is_features_normal = False
    is_labels_encod = False

    if not is_features_normal:
        train_features = normalize_grayscale(train_features)
        test_features = normalize_grayscale(test_features)
        is_features_normal = True


    if not is_labels_encod:
        # Turn labels into numbers and apply One-Hot Encoding
        encoder = LabelBinarizer()
        encoder.fit(train_labels)
        train_labels = encoder.transform(train_labels)
        test_labels = encoder.transform(test_labels)

        # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
        train_labels = train_labels.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
        is_labels_encod = True
        
        print('Labels One-Hot Encoded')


    assert is_features_normal, 'You skipped to normalize the features'
    assert is_labels_encod, 'You skipped to One-Hot Encode the labels'

    
    ## Train Valid splition
    train_features, valid_features, train_labels, valid_labels = train_test_split(train_features,
                     train_labels,
                     test_size=0.05,
                     random_state=832289)

    print('Training features and labels randomized and split.')


    ## Save the data for easy access
    pickle_file = './dataset/notMNIST.pickle'
    
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump({'train_dataset': train_features,
                            'train_labels': train_labels,
                            'valid_dataset': valid_features,
                            'valid_labels': valid_labels,
                            'test_dataset': test_features,
                            'test_labels': test_labels,},
                            pfile, pickle.HIGHEST_PROTOCOL)
        
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')

