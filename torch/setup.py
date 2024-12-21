import os
os.system("rm -rf ./torch/process_data/Music_Sliced_Images")
os.system("rm -rf ./torch/process_data/Music_Spectogram_Images")
os.system("rm -rf ./torch/process_data/Train_Sliced_Images")
os.system("rm -rf ./torch/process_data/Train_Spectogram_Images")
os.system("rm -rf ./torch/process_data/Training_Data")

#Generate spectograms for training and test data
os.system("python  ./torch/process_data/import_data.py -mTrain")
os.system("python  ./torch/process_data/import_data.py -mTest")

#Slice spectograms for training and test data
os.system("python  ./torch/process_data/slice_spectrogram.py -mTrain")
os.system("python  ./torch/process_data/slice_spectrogram.py -mTest")

#Generate numpy arrays
os.system("python  ./torch/process_data/load_train_data.py")







