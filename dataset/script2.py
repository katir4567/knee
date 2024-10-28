import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split, StratifiedKFold
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


train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)


def copy_images(dataframe, destination_folder):
    for _, row in dataframe.iterrows():
        src_path = row['fname']
        label = row['labels']
        dest_folder = os.path.join(destination_folder, label)
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)


test_folder = "data_split/Test"
for label in labels:
    os.makedirs(os.path.join(test_folder, label), exist_ok=True)


copy_images(test_df, test_folder)


def visualize_distribution(dataframe, title, filename):
    plt.figure(figsize=(8, 6))
    dataframe['labels'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.xlabel('KL Grade')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=0)
    plt.savefig(filename)
    plt.show()


visualize_distribution(test_df, 'KL Grade Distribution in Test Set', 'test_set_distribution.png')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold_num = 1
for train_index, val_index in skf.split(train_val_df, train_val_df['labels']):
    
    fold_train_df = train_val_df.iloc[train_index]
    fold_val_df = train_val_df.iloc[val_index]
    
 
    fold_train_folder = f"data_split/Fold_{fold_num}/Train"
    fold_val_folder = f"data_split/Fold_{fold_num}/Validation"
    for folder in [fold_train_folder, fold_val_folder]:
        for label in labels:
            os.makedirs(os.path.join(folder, label), exist_ok=True)

  
    copy_images(fold_train_df, fold_train_folder)
    copy_images(fold_val_df, fold_val_folder)
    
   
    visualize_distribution(fold_train_df, f'KL Grade Distribution in Training Set - Fold {fold_num}', f'train_set_distribution_fold_{fold_num}.png')
    visualize_distribution(fold_val_df, f'KL Grade Distribution in Validation Set - Fold {fold_num}', f'val_set_distribution_fold_{fold_num}.png')
    
    fold_num += 1
