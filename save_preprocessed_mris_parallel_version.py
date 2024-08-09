import os
import tempfile
import shutil
import io

from new_dicom_convert import dicom_to_nifti
from preprocess import preProcess_mri
import boto3
from io import BytesIO

import dicom2nifti
import pydicom
import nibabel as nib

import ants
import antspynet

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import  tqdm
from scipy.ndimage import zoom

from concurrent.futures import ThreadPoolExecutor


aws_access_key_id = ""
aws_secret_access_key = ""
region_name = "us-east-2"

aws_keys = {
    "access_id":aws_access_key_id,
    "secret_key":aws_secret_access_key,
    "region":region_name
}
s3_bucket_name = 'my-hosted-mris'

s3_client = boto3.client('s3',
                            aws_access_key_id=aws_keys["access_id"],
                            aws_secret_access_key=aws_keys["secret_key"],
                            region_name=aws_keys["region"]
                            )
s3_resource_upload = boto3.resource('s3',
                                      aws_access_key_id=aws_keys["access_id"],
                                      aws_secret_access_key=aws_keys["secret_key"],
                                      region_name=aws_keys["region"]
                                      )
def _dicom_to_nifti(dicom_files):
  nifti_img = dicom_to_nifti(dicom_files,None,reorient_nifti =  True)
  return nifti_img['NII'].get_fdata()

def _download_dicom_file(s3_client,s3_bucket_name, s3_key):
  obj = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_key)
  dicom_bytes = obj['Body'].read()
  dicom_file = pydicom.dcmread(BytesIO(dicom_bytes))
  return dicom_file

def _resize_scipy_image(nifti_data):
      target_shape = (128, 128, 100)
      resize_factors = [
          target_size / original_size for target_size, original_size in zip(target_shape, nifti_data.shape)
      ]
      resized_data = zoom(nifti_data, resize_factors, order=1)  # Use order=1 for bilinear interpolation
      # resized_nifti_image_scipy = nib.Nifti1Image(resized_data, affine=out.affine)
      # nib.save(resized_nifti_image, 'resized_image.nii.gz')
      return resized_data


def _save_as_npz(s3_resource_upload,nii_array,s3_prefix):
  try:
   
    bucket = s3_resource_upload.Bucket(s3_bucket_name)
    out_file = io.BytesIO()
    np.savez_compressed(out_file, nii_array)
    out_file.seek(0)
    s3_file_emb = f'{s3_prefix}.npz'
    bucket.put_object(Key=s3_file_emb, Body=out_file) 
    return "1"
  except Exception as e:
    print(e)
    return "0"

def _get_file_list(s3_client,s3_bucket_name,s3_prefix):
  response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
  file_list = [obj['Key'] for obj in response.get('Contents', [])]
  return file_list

def process_mri_image(file_path, s3_client, s3_resource_upload,s3_bucket_name):
  dicom_files = []
  for fi in _get_file_list(s3_client,s3_bucket_name,file_path):
      dicom_file = _download_dicom_file(s3_client,s3_bucket_name,fi)
      dicom_files.append(dicom_file)
  nifti_array = _dicom_to_nifti(dicom_files)
  resized_nifti_array = _resize_scipy_image(nifti_array)
  ants_image = ants.ants_image_io.from_numpy(nifti_array)
  segmented_img = preProcess_mri(ants_image)
  output = segmented_img.numpy()
  s3_prefix =  "processed_mris/normal/"+file_path.replace("/","_").replace("All_MRIs_","")
  s3_prefix_preprocessed =  "processed_mris/preprocessed/"+file_path.replace("/","_").replace("All_MRIs_","")
  normal_file_status = _save_as_npz(s3_resource_upload,resized_nifti_array,s3_prefix)
  processed_file_status = _save_as_npz(s3_resource_upload,output,s3_prefix_preprocessed)

  return normal_file_status, processed_file_status

executor = ThreadPoolExecutor(max_workers=6)

df = pd.read_csv("image_paths_along_with_labels_part_a.csv").tail(10)
file_name_tracker = df['files'].to_list()
labels_tracker = df['labels'].to_list()

futures = []
for file_path in file_name_tracker:
  future = executor.submit(process_mri_image, file_path, s3_client,s3_resource_upload, s3_bucket_name)
  futures.append(future)

normal_file_status_tracker = []
processed_file_status_tracker = []
for future in futures:
  normal_file_status, processed_file_status = future.result()
  normal_file_status_tracker.append(normal_file_status)
  processed_file_status_tracker.append(processed_file_status)

new_df = pd.DataFrame({"files":file_name_tracker,
                       "labels":labels_tracker,
                       "normal_file_status":normal_file_status_tracker,
                       "processed_file_status": processed_file_status_tracker
                       })

new_df.to_csv("final_meta_saved.csv",index= False)
