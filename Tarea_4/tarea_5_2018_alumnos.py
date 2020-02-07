#####################################################################################################
# testing VGG face model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
#####################################################################################################

from vgg_face import vgg_face
#from scipy.misc import imread, imresize#, imwrite
from imageio import imread, imwrite
from skimage.transform import resize
import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler


# Image Paths 
img_paths = []
rootdir = './separated'
for race in sorted(os.listdir(rootdir)):
    filenames = sorted(os.listdir(rootdir + '/' + race))
    filenames = list(map(lambda filename: rootdir + '/' + race + '/' + filename, filenames))
    img_paths.append(filenames)

# Feature Vector Extraction
layers = ['relu6', 'relu7']
for layer in layers:
    feature_vec = []
    # Graph Building
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face('vgg-face.mat', input_maps)
        feats = output[layer]
    # Session Run
    with tf.Session(graph=graph) as sess:
        print('\nFeature Vector: ' + layer)
        for race_paths in img_paths:
            race_feature_vec = []
            for img_path in race_paths:
                img = imread(img_path, pilmode='RGB');
                img = img[0:250, :, :]
                img = resize(img, [224, 224])  #imresize
                img = img - average_image
                # Feature Extraction
                [feat_vals] = sess.run([feats], feed_dict={input_maps: [img]})
                feat_vals = np.reshape(feat_vals, (1, 4096))
                race_feature_vec.append(feat_vals)
            feature_vec.append(race_feature_vec)
            print(race_paths[0].split('/')[-2] + ' Feature Vector Generated!')
    # Features Saving
    feature_vec = np.squeeze(np.array(feature_vec))
    np.save('./' + layer, feature_vec)
    print('Feature vector ' + layer + ' saved sucessfully!\n')