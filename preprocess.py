import antspynet
import ants
import time

def segmentation(img):
    """
    img is of type ants
    """
    seg = ants.kmeans_segmentation(img, 3)
    seg['segmentation'].plot(title = 'Segmentation')
    return seg['segmentation']

def n4BiasFieldCorrection(img):
    """
    image is of type ants
    """
    #img.plot(title = 'Pre N4 Bias Field Correction')
    img_n4 = ants.n4_bias_field_correction(img)
    #img_n4.plot(title = 'N4 Bias Field Corrected')
    return img_n4

def brainExtract(image):
    """
    image is of type ants
    Function will also normalize intensities aka rescale intensities to between 0 and 1
    """
    probability_brain_mask = antspynet.utilities.preprocess_brain_image(image, brain_extraction_modality="t1", do_bias_correction=True, intensity_normalization_type = "01")
    brain = probability_brain_mask['brain_mask']*image
    #brain.plot(title='Brain')
    return brain

def registration(ant_Moving, fixed):
    moving = ants.resample_image(ant_Moving, (128, 128, 100), use_voxels = True, interp_type = 0) #256, 256, 160
    #fixed.plot(overlay = moving, title = 'Before Registration')
    mytx = ants.registration(fixed = fixed, moving = moving, type_of_transform = 'SyN')['warpedmovout']
    #warped_moving = mytx
    #fixed.plot(overlay=warped_moving, title='After Registration')
    return mytx

def preProcess_mri(Image):
    fixed = ants.resample_image(Image, (128, 128, 80), use_voxels=True, interp_type=0)  # interp+type = 0 means linear; 256, 256, 160
    registered_Img = registration(Image, fixed)
    brain = brainExtract(registered_Img)
    brainN4 = n4BiasFieldCorrection(brain)
    segmentedMRI = segmentation(brainN4)
    return segmentedMRI
