import os
import re
import argparse
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from google.cloud import storage

def get_gcs_client():
    return storage.Client()

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")

def create_spectrogram(args):
    mode = args.mode
    run_on_gcp = args.run_on_gcp.lower() == 'true'

    if run_on_gcp:
        bucket_name = "your-gcs-bucket-name"  # Replace with your GCS bucket name
    else:
        bucket_name = None

    if mode == "Train":
        train_spectrogram_dir = "torch/process_data/Train_Spectogram_Images"
        if not os.path.exists(train_spectrogram_dir):
            os.makedirs(train_spectrogram_dir)

        if run_on_gcp:
            metadata_blob = "torch/process_data/fma_metadata/tracks.csv"
            metadata_local = "tracks.csv"
            download_from_gcs(bucket_name, metadata_blob, metadata_local)
        else:
            metadata_local = "torch/process_data/fma_metadata/tracks.csv"

        tracks = pd.read_csv(metadata_local, header=2, low_memory=False)
        tracks_array = tracks.values
        tracks_id_array = tracks_array[:, 0]
        tracks_genre_array = tracks_array[:, 40]
        tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
        tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

        if run_on_gcp:
            folder_sample = "torch/process_data/fma_small/"
            audio_blobs = get_gcs_client().list_blobs(bucket_name, prefix=folder_sample)
        else:
            folder_sample = "torch/process_data/fma_small"
            directories = [d for d in os.listdir(folder_sample) if os.path.isdir(os.path.join(folder_sample, d))]

        counter = 0
        print("Converting mp3 audio files into mel spectrograms ...")
        if run_on_gcp:
            for blob in audio_blobs:
                if blob.name.endswith(".mp3"):
                    local_mp3 = os.path.basename(blob.name)
                    download_from_gcs(bucket_name, blob.name, local_mp3)
                    process_mp3(local_mp3, tracks_id_array, tracks_genre_array, train_spectrogram_dir, counter, bucket_name, run_on_gcp)
                    counter += 1
        else:
            for d in directories:
                label_directory = os.path.join(folder_sample, d)
                file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".mp3")]
                for f in file_names:
                    process_mp3(f, tracks_id_array, tracks_genre_array, train_spectrogram_dir, counter, bucket_name, run_on_gcp)
                    counter += 1
        return

    elif mode == "Test":
        test_spectrogram_dir = "torch/process_data/Music_Spectogram_Images"
        if not os.path.exists(test_spectrogram_dir):
            os.makedirs(test_spectrogram_dir)

        if run_on_gcp:
            folder_sample = "torch/templates/music/"
            music_blobs = get_gcs_client().list_blobs(bucket_name, prefix=folder_sample)
        else:
            folder_sample = "torch/templates/music"
            file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample) if f.endswith(".mp3")]

        counter = 0
        print("Converting mp3 audio files into mel spectrograms ...")
        if run_on_gcp:
            for blob in music_blobs:
                if blob.name.endswith(".mp3"):
                    local_mp3 = os.path.basename(blob.name)
                    download_from_gcs(bucket_name, blob.name, local_mp3)
                    process_mp3(local_mp3, None, None, test_spectrogram_dir, counter, bucket_name, run_on_gcp, test_id=True)
                    counter += 1
        else:
            for f in file_names:
                process_mp3(f, None, None, test_spectrogram_dir, counter, bucket_name, run_on_gcp, test_id=True)
                counter += 1
        return

def process_mp3(file_path, tracks_id_array, tracks_genre_array, output_dir, counter, bucket_name, run_on_gcp, test_id=False):
    try:
        if not test_id:
            track_id = int(re.search(r'fma_small/.*/(.+?).mp3', file_path).group(1))
            track_index = list(tracks_id_array).index(track_id)
            if str(tracks_genre_array[track_index, 0]) == '0':
                return

        y, sr = librosa.load(file_path)
        melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel = librosa.power_to_db(melspectrogram_array)

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = float(mel.shape[1]) / 100
        fig_size[1] = float(mel.shape[0]) / 100
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis('off')
        plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(mel, cmap='gray_r')

        if test_id:
            test_id_str = re.search(r'music/(.+?).mp3', file_path).group(1)
            spectrogram_path = f"{output_dir}/{test_id_str}.jpg"
        else:
            spectrogram_path = f"{output_dir}/{counter}_{tracks_genre_array[track_index, 0]}.jpg"

        plt.savefig(spectrogram_path, dpi=100)
        plt.close()

        if run_on_gcp:
            upload_to_gcs(bucket_name, spectrogram_path, f"{output_dir}/{os.path.basename(spectrogram_path)}")
    except Exception as e:
        print(f"An exception occurred: {e}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Converts the files mp3 into mel-spectrograms')
    argparser.add_argument('-m', '--mode', required=True, help='set mode to process data for Train or Test')
    argparser.add_argument('-g', '--run_on_gcp', required=True, help='set to True to run on GCP, False to run locally')
    args = argparser.parse_args()
    create_spectrogram(args)
