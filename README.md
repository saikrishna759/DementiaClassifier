### DementiaClassifier

## iteration 1
mri_dataset.py is dataloading code

new_dicom_convert.py and image_reoirentation2.py are the custom modified files of dicom2niftii library( Made some internal changes in dicom2niftii library for our usecase) 

image_paths_along_with_labels.csv contains all the file patients mri image file paths along with their type(label) {note: only files uploaded till now}

## iteration 2

Added regular preprocess step, preprocess.py (which performs resizing (128,128,100), registration, brain extraction, denoising, segmentation)

#Time taken stats <br>
 
downloading 160 dicom images : 20 sec (based on my internet speed) <br>
dataloading before preprocessing[4 per batch] : 110 sec (30 sec for each point) <br>
saving resized data before preprocessing takes : 6mb <br>

registration: 8.705140590667725 <br>
brain extractiton: 30.245099544525146 <br>
bias correction: 2.585256814956665 <br>
segmentation: 9.691402196884155 <br>

dataloading before preprocessing[4 per batch] : 514 sec (120 sec for each point) <br>
saving preprocessed data as npz compressed files takes : 200kb <br>


suggestion : To experiment multiple times on data, it is better to save the preprocessed data and use it for faster training







