import os
import re
import cv2
import numpy as np 
import argparse
from sklearn.model_selection import train_test_split
import sys,os
sys.path.append(os.getcwd())

def load_dataset(args):
    datasetSize = args.size 
    genre = {
    "Hip-Hop": 0,
    "International": 1,
    "Electronic": 2,
    "Folk" : 3,
    "Experimental": 4,
    "Rock": 5,
    "Pop": 6,
    "Instrumental": 7
    }
    training_data_path = "/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data"
    train_sliced_images_path = "/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Train_Sliced_Images"
    regex_train_slcied_images_path = r'/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Train_Sliced_Images/(.+?)_.*.jpg'

    print("Compiling Training and Testing Sets ...")
    filenames = [('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Train_Sliced_Images', f) for f in os.listdir('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Train_Sliced_Images')
                    if f.endswith(".jpg")]

    images_all = [None]*(len(filenames))
    labels_all = [None]*(len(filenames))
    for f in filenames:
        index = int(f[1].split('_')[0])
        genre_variable = f[1].split('_')[-1].replace('.jpg', '')
        file_name_full = f[0] + '/' + f[1]
        temp = cv2.imread(file_name_full, cv2.IMREAD_UNCHANGED)
        images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        labels_all[index] = genre[genre_variable]    
    if(datasetSize == 1.0):
        images = images_all
        labels = labels_all

    else:
        count_max = int(len(images_all)*datasetSize / 8.0)
        count_array = [0, 0, 0, 0, 0 ,0, 0, 0]
        images = []
        labels = []
        for i in range(0, len(images_all)):
            if(count_array[labels_all[i]] < count_max):
                images.append(images_all[i])
                labels.append(labels_all[i])
                count_array[labels_all[i]] += 1
        images = np.array(images)
        labels = np.array(labels)

    images = np.array(images)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0],1)
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.20, shuffle=True)
    n_classes = len(genre)
    genre_new = {value: key for key, value in genre.items()}
    if not os.path.exists('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data'):
        os.makedirs('/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data')
    np.save("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/train_x.npy", train_x)
    np.save("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/train_y.npy", train_y)
    np.save("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_x.npy", test_x)
    np.save("/Users/wajidabdul/courses/cis579/code/muze/torch/process_data/Training_Data/test_y.npy", test_y)
    return train_x, train_y, test_x, test_y, n_classes, genre_new


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Imports images, converts them into grayscale, and then exports them as numpy matrices for training and testing')
    argparser.add_argument('-s', '--size', type=float, default= '1.0', help='set datasize process for training data')
    args = argparser.parse_args()
    load_dataset(args)