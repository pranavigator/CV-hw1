from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    # # Q1.1
    #img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg') 
    # img_path = join(opts.data_dir, 'kitchen/sun_aaslbwtcdcwjukuo.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255

    # # #Normalizing image
    # # img = img - np.min(img)
    # # img = img / np.max(img)

    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    print("Dictionary Creation complete")

    # Q1.3
    # img_path = join(opts.data_dir, 'aquarium\sun_aadolwejqiytvyne.jpg')
    # img_path = join(opts.data_dir, 'laundromat/sun_aaxufyiupegixznm.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)

    # Q2.1-2.4

    # Q2.1 Test
    # train_files = open(join(opts.data_dir, "train_files.txt")).read().splitlines()
    # train_labels = np.loadtxt(join(opts.data_dir, "train_labels.txt"), np.int32)
    # dictionary = np.load(join(opts.out_dir, "dictionary.npy"))

    # img_path = join(opts.data_dir, 'aquarium\sun_aadolwejqiytvyne.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # print(hist)

    # Q2.2 Test
    # img_path = join(opts.data_dir, 'aquarium\sun_aadolwejqiytvyne.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # hist_all = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)

    #Q2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    print("Recognition system built")

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
