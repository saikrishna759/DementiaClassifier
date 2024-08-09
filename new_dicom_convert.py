from nibabel.pydicom_compat import pydicom
from pydicom.tag import Tag

import dicom2nifti.common as common
import dicom2nifti.convert_ge as convert_ge
import dicom2nifti.convert_generic as convert_generic
import dicom2nifti.convert_philips as convert_philips
import dicom2nifti.convert_siemens as convert_siemens
import dicom2nifti.resample as resample
import dicom2nifti.settings as settings
from dicom2nifti import convert_hitachi
from dicom2nifti.exceptions import ConversionValidationError, ConversionError
import dicom2nifti.convert_dicom as cd
from image_reoirentation2 import reorient_image

def dicom_to_nifti(dicom_list,output_file,reorient_nifti):
  vendor = cd._get_vendor(dicom_list)
  if vendor == cd.Vendor.GENERIC:
      results = convert_generic.dicom_to_nifti(dicom_list, output_file)
  elif vendor == cd.Vendor.SIEMENS:
      results = convert_siemens.dicom_to_nifti(dicom_list, output_file)
  elif vendor == cd.Vendor.GE:
      results = convert_ge.dicom_to_nifti(dicom_list, output_file)
  elif vendor == cd.Vendor.PHILIPS:
      results = convert_philips.dicom_to_nifti(dicom_list, output_file)
  elif vendor == cd.Vendor.HITACHI:
      results = convert_hitachi.dicom_to_nifti(dicom_list, output_file)
  else:
      raise ConversionValidationError("UNSUPPORTED_DATA")

  # do image reorientation if needed
  if reorient_nifti or settings.resample:
      results['NII'] = reorient_image(results['NII'])

  # resampling needs to be after reorientation
  if settings.resample:
      if not common.is_orthogonal_nifti(results['NII']):
          results['NII'] = resample.resample_single_nifti(results['NII'], results['NII_FILE'])

  return results