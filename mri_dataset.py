import os
import tempfile
import shutil

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


def mri_plot(nifti_data,slice_index = None):
  # nifti_data = img.get_fdata()
  if slice_index is None or slice_index >= 166:
    slice_index = nifti_data.shape[2] // 2  # Displaying the middle slice
  plt.imshow(nifti_data[:, :, slice_index], cmap='gray')
  plt.axis('off')
  plt.show()


class DICOMToNiftiDataset(Dataset):
    def __init__(self, s3_bucket_name, aws_keys,data_file_path):
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = boto3.client('s3',
                                      aws_access_key_id=aws_keys["access_id"],
                                      aws_secret_access_key=aws_keys["secret_key"],
                                      region_name=aws_keys["region"]
                                      )
        self.data_file = data_file_path
        self.df = self._read_dataframe()

    def _read_dataframe(self):
        df_all = pd.read_csv(self.data_file)
        new_df = df_all.groupby('labels', group_keys=False).apply(lambda x : x.sample(min(100, len(x))))
        return new_df

    def _resize_scipy_image(self,nifti_data):
        target_shape = (128, 128, 100)
        resize_factors = [
            target_size / original_size for target_size, original_size in zip(target_shape, nifti_data.shape)
        ]
        resized_data = zoom(nifti_data, resize_factors, order=1)  # Use order=1 for bilinear interpolation
        # resized_nifti_image_scipy = nib.Nifti1Image(resized_data, affine=out.affine)
        # nib.save(resized_nifti_image, 'resized_image.nii.gz')
        return resized_data

    def _resize_torch_image(self,nifti_data):
        target_shape = (100, 100, 166)
        nifti_tensor = torch.from_numpy(nifti_data).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(nifti_tensor, size=target_shape, mode='trilinear', align_corners=True)
        resized_data = resized_tensor.squeeze().numpy()
        # resized_nifti_image = nib.Nifti1Image(resized_data, affine=out.affine)
        return resized_data

    def _get_file_list(self,s3_prefix):
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=s3_prefix)
        file_list = [obj['Key'] for obj in response.get('Contents', [])]
        return file_list

    def _download_dicom_file(self, s3_key):
        obj = self.s3_client.get_object(Bucket=self.s3_bucket_name, Key=s3_key)
        dicom_bytes = obj['Body'].read()
        dicom_file = pydicom.dcmread(BytesIO(dicom_bytes))
        return dicom_file

    def _dicom_to_nifti(self, dicom_files):
        nifti_img = dicom_to_nifti(dicom_files,None,reorient_nifti =  True)
        return nifti_img['NII'].get_fdata()

    

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        file_path = data['files']
        label = data['labels']
        dicom_files = []
        for fi in self._get_file_list(file_path):
            dicom_file = self._download_dicom_file(fi)
            dicom_files.append(dicom_file)
        nifti_array = self._dicom_to_nifti(dicom_files)
        # resized_nifti_array = self._resize_scipy_image(nifti_array)
        ants_image = ants.ants_image_io.from_numpy(nifti_array)
        segmented_img = preProcess_mri(ants_image)
        # return segmented_img.numpy(),np.array([label])
        return torch.tensor(segmented_img.numpy(), dtype=torch.float32),torch.tensor(np.array([label]), dtype=torch.float32)



# code testing

csv_file_path = "image_paths_along_with_labels.csv"

aws_access_key_id = ""
aws_secret_access_key = ""
region_name = "us-east-2"

aws_keys = {
    "access_id":aws_access_key_id,
    "secret_key":aws_secret_access_key,
    "region":region_name
}
s3_bucket_name = 'my-hosted-mris'

dataset = DICOMToNiftiDataset(s3_bucket_name, aws_keys)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(len(dataloader))

iter_dataloader = iter(dataloader)

import time
t1 = time.time()
batch_data = next(iter_dataloader)
t2 = time.time()
print(t2-t1)

#plot test sample
mri_plot(batch_data[0][0].numpy())

np.savez_compressed("sample.npz",batch_data[0].numpy())