# AUTO-WCEBleedGen Challenge

Automatic detection and classification of bleeding and non-bleeding frames from Wireless Capsule Endoscopy (WCE) videos, for the WCEBleedGen challenge.

## What is in this repo

| Folder / file | Contents |
|---|---|
| `Bleeding & Non-Bleeding Classification/` | Notebook `Bleeding_&_Non_Bleeding_classification.ipynb` and the training log `log_classification.csv` |
| `Bleeding & Non-Bleeding Segmentation/` | Notebook `Bleeding_&_Non_Bleeding_Segmentation.ipynb` and the training log `log_segmentation.csv` |
| `Dataset/` | `Classification Dataframe.csv` |
| `Results/` | Predicted classifications (`Bleeding & Non-Bleeding Classification( Predicted ).csv`), segmented images and bounding box images with confidence levels, for Test Dataset 1 and Test Dataset 2 |

## Classification notebook

- Reads image paths and labels from a CSV, loads the images resized to 224x224x3 and splits them 80/20.
- Trains a convolutional neural network built from stacked `Conv2D` layers followed by dense layers (64, 32, 16) and a single sigmoid output for bleeding versus non-bleeding.
- Uses `EarlyStopping` and `CSVLogger` callbacks, and evaluates with the F1 score.

## Segmentation notebook

- Pairs each bleeding image with its annotation mask from the training set.
- Trains an image segmentation network built from `Conv2D`, `MaxPooling2D`, `Conv2DTranspose` and `Concatenate` layers, using the Adam optimiser with `EarlyStopping`, `ModelCheckpoint` and `CSVLogger` callbacks.

## Run

Both notebooks were written for Google Colab. They read the challenge dataset from Google Drive paths such as `/content/drive/MyDrive/1 WCEBleedGen Challenge/`, so mount your own Drive copy of the dataset or change the paths. The challenge images themselves are not in this repo.

## Tech stack

Python, TensorFlow / Keras, OpenCV, scikit-learn, pandas, NumPy, matplotlib.
