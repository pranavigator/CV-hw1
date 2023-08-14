import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
from sklearn import preprocessing as prep

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    # ----- TODO -----
    hist, bin_edges = np.histogram(wordmap, bins = range(K+1), density = True)
    hist = hist / np.sum(hist) #Normalizing histogram

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L
    # ----- TODO -----
    weights = np.empty(L+1)
    for l_w in range(len(weights)):
        if l_w <= 1:
            weights[l_w] = 2**(-L)
        else:
            weights[l_w] = 2**(l_w - L - 1)

    count = 0 #counter variable for hist_all array creation

    for l_hist in range(L, -1, -1):
        cell_split = 2**l_hist
        block_split_row = int(np.ceil(wordmap.shape[0] / cell_split)) #splitting wordmap across rows
        block_split_col = int(np.ceil(wordmap.shape[1] / cell_split)) #splitting wordmap across columns
        for i in range(0, wordmap.shape[0], block_split_row):
            for j in range(0, wordmap.shape[1], block_split_col):
                block = wordmap[i:i + block_split_row, j:j + block_split_col]
                hist_block = get_feature_from_wordmap(opts, block) * weights[l_hist]
                hist_block = np.reshape(hist_block, (K,1))
                if i == 0 and j == 0 and count == 0:
                    hist_all = hist_block
                    count = 1
                else: 
                    hist_all = np.vstack((hist_all, hist_block)) #appending the block histograms to hist_all
                
    
    hist_all = hist_all / np.sum(hist_all) #Normalizing full histogram

    return hist_all


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # ----- TODO -----
    for i in range(len(train_files)):
        img_path = join(opts.data_dir, train_files[i])
        if i == 0:
            features = get_image_feature(opts, img_path, dictionary)
        else:
            features = np.hstack((features, get_image_feature(opts, img_path, dictionary)))
        if i % 50 == 0:
            print("Image number:", i)

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                             features=features,
                             labels=train_labels,
                             dictionary=dictionary,
                             SPM_layer_num=SPM_layer_num,
                       ) 

    pass

    # example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )


def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * dist: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    sim = np.sum(np.minimum(word_hist.flatten(), histograms.T), axis = 1) #axis = 1
    dist = 1 - sim #Distance is the inverse of similarity

    return dist


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"), allow_pickle = True)
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    # ----- TODO -----
    train_features = trained_system["features"]
    train_labels = trained_system["labels"]
    confusion_matrix = np.zeros((8,8))

    for i in range(len(test_files)):
        if i % 50 == 0:
            print("Image number:", i)

        img_path = join(opts.data_dir, test_files[i])
        feature = get_image_feature(opts, img_path, dictionary)
        test_dist = distance_to_set(feature, train_features)
        predict_label = train_labels[np.argmin(test_dist)]
        # print("Predict Label:", predict_label)
        # print("Actual Label:", test_labels[i])
        confusion_matrix[test_labels[i], predict_label] = confusion_matrix[test_labels[i], predict_label] + 1

    sys_accuracy = (np.trace(confusion_matrix)/np.sum(confusion_matrix)) * 100
    
    return confusion_matrix, sys_accuracy
