import io
import os

import torch
from google.cloud import storage

storage_client = None


def save_to_local(state_dict, output_dir: str, file_name: str):
    torch.save(state_dict, os.path.join(output_dir, file_name))


def load_from_local(output_dir: str, file_name: str):
    return torch.load(os.path.join(output_dir, file_name))


def save_to_gcs(state_dict, output_dir: str, file_name: str):
    global storage_client

    if not storage_client:
        storage_client = storage.Client()

    output_bucket = storage_client.get_bucket("pandl_1")
    dir = output_dir.split("/")[-1]

    src_file_path = os.path.join(f"/tmp/{dir}", file_name)
    blob = output_bucket.blob(f"out/{dir}/{file_name}")
    blob.upload_from_filename(src_file_path)

    torch.save(state_dict, os.path.join(f"/tmp/{dir}", file_name))


def load_from_gcs(output_dir: str, file_name: str):
    global storage_client

    if not storage_client:
        storage_client = storage.Client()
    output_bucket = storage_client.get_bucket("pandl_1")
    dir = output_dir.split("/")[-1]
    model_blob = output_bucket.get_blob(f"out/{dir}/{file_name}")
    model_blob = model_blob.download_as_string()
    model_buffer = io.BytesIO(model_blob)
    return torch.load(model_buffer)
