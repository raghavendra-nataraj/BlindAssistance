Files:
Viola_Jones : This folder contains training images, model and execution for outdoor door detection.
 |- model : Model cascade.xml
 |- outdoor_detect_doors.py: Executable
 |- model_train_files : Model train data
    |- Positive Samples: Cropped door images
    |- Negative Samples: background images
    |- vec File: Sampled features for Viola-Jones classifier


How to execute:

Outdoor door detection: python outdoor_detect_doors.py

