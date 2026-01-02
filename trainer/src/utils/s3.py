import boto3
import os

def upload_to_s3(local_path: str, s3_key: str):
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )
    bucket = os.environ["S3_BUCKET_NAME"]

    s3.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=s3_key,
    )

