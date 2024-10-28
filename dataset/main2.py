import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the base path to your dataset
dataset_path = 'dataset/Labeled'  # Change this to your dataset path
labels = ["0", "1", "2", "3", "4"]  # Define the class labels (subfolder names)

# Step 1: Create a DataFrame to Store Image Paths and Labels
image_data = []

# Traverse through each class folder and collect image paths
for label in labels:
    label_folder = os.path.join(dataset_path, label)  # Get the path to the label folder
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        if os.path.isfile(image_path):  # Check if it is a file
            image_data.append((image_path, label))

# Convert the image data into a DataFrame
df = pd.DataFrame(image_data, columns=["fname", "labels"])

# Step 2: Split the Dataset into Train, Validation, and Test Sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)

# Display the number of images in each set
print(f"Number of images in Train Set: {len(train_df)}")
print(f"Number of images in Validation Set: {len(val_df)}")
print(f"Number of images in Test Set: {len(test_df)}")

# Step 3: Create Destination Folders for Train, Validation, and Test Sets
train_folder = "dataset/data_split/Train"
validation_folder = "dataset/data_split/Validation"
test_folder = "dataset/data_split/Test"

# Create the destination folders and their subfolders if they don't exist
for folder in [train_folder, validation_folder, test_folder]:
    for label in labels:
        os.makedirs(os.path.join(folder, label), exist_ok=True)

# Step 4: Define a Function to Copy Images to the Appropriate Folders
def copy_images(dataframe, destination_folder):
    """
    Copies images to the specified destination folder.
    Arguments:
    - dataframe: DataFrame containing the image paths and labels
    - destination_folder: The folder to copy the images to
    """
    for _, row in dataframe.iterrows():
        src_path = row['fname']
        label = row['labels']
    
