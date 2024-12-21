import zipfile
from google.cloud import storage

def unzip_file_in_gcp(project_id, bucket_name, zip_file_path, extract_to_path):
    """Unzips a file in a GCS bucket."""

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(zip_file_path)

    with blob.open("rb") as zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)

# Example usage:
project_id= 'muze-v2'
bucket_name = 'fma_medium_dataset'
zip_file_path = 'os.unil.cloud.switch.ch/fma/fma_medium.zip'
extract_to_path = 'os.unil.cloud.switch.ch/fma/fma_medium-decompressed'

unzip_file_in_gcp(project_id, bucket_name, zip_file_path, extract_to_path)