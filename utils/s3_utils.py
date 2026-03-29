import boto3
import os

def upload_to_s3(file_path, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket
    
    :param file_path: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_path is used
    :return: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client('s3')
    
    try:
        print(f"Uploading {file_path} to s3://{bucket_name}/{object_name}")
        s3_client.upload_file(file_path, bucket_name, object_name)
        print("Upload successful!")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False
