import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from multiprocessing import Pool
import scipy.spatial.distance

def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img     : numpy.ndarray of shape (H,W) or (H,W,3) or (H,W,4) with range [0, 1]
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales
    # ----- TODO -----    
    
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis = 2).repeat(3, axis=2)
    
    elif img.shape[2] > 3:
        img = np.delete(img, 3, axis = 2)

   
    img_ext = skimage.color.rgb2lab(img)
    filter_responses = np.zeros((img_ext.shape[0], img_ext.shape[1], 3 * 4 * len(filter_scales))) #3*F where F is 4 * len(filter_scales)

    for i, sigma in enumerate(filter_scales):
        for j in range(3):
            filter_responses[:, :, 3 * 4 * i + j] = scipy.ndimage.gaussian_filter(img_ext[:, :, j], sigma)
            filter_responses[:, :, 3 * 4 * i + j + 3] = scipy.ndimage.gaussian_laplace(img_ext[:, :, j], sigma)

            #Derivatives
            filter_responses[:, :, 3 * 4 * i + j + 6] = scipy.ndimage.gaussian_filter(img_ext[:, :, j], sigma, [0, 1]) #x direction
            filter_responses[:, :, 3 * 4 * i + j + 9] = scipy.ndimage.gaussian_filter(img_ext[:, :, j], sigma, [1, 0]) #y direction
    
    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    # Did not use. Unable to get working
    # filter_responses, feat_dir, alpha = args
    # print(filter_responses.shape)
    

    pass


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    # ----- TODO -----
    filter_scales = opts.filter_scales
    response_ext = np.empty((alpha * len(train_files), 3 * 4 *len(filter_scales))) #(alpha * T, 3 * F)

    #Could not get to work due to complications with zipping iterables (extra credit attempt)
    #pool = multiprocessing.Pool(processes = n_worker * 4)
    #args = zip(filter_responses, feat_dir, alpha)
    #pool.map(compute_dictionary_one_image, args)

    for j in range(len(train_files)):
        img_path = join(data_dir, train_files[j])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32) / 255 #Normalizing image
        filter_responses = extract_filter_responses(opts, img)

        for k in range(alpha):
            row = np.random.randint(0, filter_responses.shape[0] - 1)
            column = np.random.randint(0, filter_responses.shape[1] - 1)
            response_ext[k * j,:] = filter_responses[row, column, :]

        if j % 50 == 0:
            print("Image Number: ", j)
        
    
    kmeans = KMeans(n_clusters=K).fit(response_ext) #KMeans clustering
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    pass

    # example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img)
    wordmap = np.empty((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = np.array([filter_responses[i, j, :]])
            euclid_dist = scipy.spatial.distance.cdist(pixel, dictionary)
            wordmap[i, j] = np.argmin(euclid_dist) #not sure whether min or argmin

    return wordmap
