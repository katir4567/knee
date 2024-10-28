import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset_path = 'dataset/Labeled'  
labels = ["0", "1", "2", "3", "4"]  


image_data = []


for label in labels:
    label_folder = os.path.join(dataset_path, label)  
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        if os.path.isfile(image_path):  
            image_data.append((image_path, label))


df = pd.DataFrame(image_data, columns=["fname", "labels"])


#train_df, test_df = train_test_split(df, test_size=0.8, stratify=df['labels'], random_state=42)
#train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=42)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['labels'], random_state=42)


print(f"Number of images in Train Set: {len(train_df)}")
print(f"Number of images in Validation Set: {len(val_df)}")
print(f"Number of images in Test Set: {len(test_df)}")


train_folder = "data_split/Train"
validation_folder = "data_split/Validation"
test_folder = "data_split/Test"


for folder in [train_folder, validation_folder, test_folder]:
    for label in labels:
        os.makedirs(os.path.join(folder, label), exist_ok=True)


def copy_images(dataframe, destination_folder):
    
    for _, row in dataframe.iterrows():
        src_path = row['fname']
        label = row['labels']
        dest_folder = os.path.join(destination_folder, label)  
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))  
        shutil.copy(src_path, dest_path) 


copy_images(train_df, train_folder)
copy_images(val_df, validation_folder)
copy_images(test_df, test_folder)


def visualize_distribution(dataframe, title, filename):
    """
    Visualizes the distribution of KL grades in the given DataFrame.
    Arguments:
    - dataframe: DataFrame containing the image labels
    - title: Title for the plot
    - filename: Filename to save the plot image
    """
    plt.figure(figsize=(8, 6))
    dataframe['labels'].value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel('KL Grade')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=0)
    plt.savefig(filename)
    plt.show()

visualize_distribution(df, 'KL Grade Distribution in Entire Dataset', 'entire_dataset_distribution.png')


visualize_distribution(train_df, 'KL Grade Distribution in Training Set', 'train_set_distribution.png')
visualize_distribution(val_df, 'KL Grade Distribution in Validation Set', 'validation_set_distribution.png')
visualize_distribution(test_df, 'KL Grade Distribution in Test Set', 'test_set_distribution.png')
