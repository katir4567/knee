import os
import shutil
import random
import matplotlib.pyplot as plt


main_folder = "dataset/Labeled"   # Main folder containing subfolders for each category 
train_folder = "dataset/Train"
validation_folder = "dataset/Validation"
test_folder = "dataset/Test"


# Define how much data we want in each set
train_ratio = 0.8         # 70% of the data will go to training
validation_ratio = 0.1   # 15% will go to validation
test_ratio = 0.1         # 15% will go to testing

# Go through each label folder  inside the main folder
for label in os.listdir(main_folder):
    label_path = os.path.join(main_folder, label)  # Path to each subfolder (e.g., "dataset/Labeled/cats")

    if os.path.isdir(label_path):
        images = os.listdir(label_path)  # List all images in that folder
        random.shuffle(images)  # Shuffle the images to get a random order

        # Split the data based on the ratios
        train_size = int(train_ratio * len(images))
        validation_size = int(validation_ratio * len(images))
        
        train_images = images[:train_size]
        validation_images = images[train_size:train_size + validation_size]
        test_images = images[train_size + validation_size:]

        # Function to copy images to the new folders
        def copy_images(image_list, destination_folder):
            dest_label_folder = os.path.join(destination_folder, label)
            if not os.path.exists(dest_label_folder):
                os.makedirs(dest_label_folder)
            for image in image_list:
                src = os.path.join(label_path, image)
                dest = os.path.join(dest_label_folder, image)
                shutil.copy(src, dest)

        # Copy the images into Train, Validation, and Test folders
        copy_images(train_images, train_folder)
        copy_images(validation_images, validation_folder)
        copy_images(test_images, test_folder)

# Visualize the number of images in each set
def visualize_data_distribution():
    set_folders = [train_folder, validation_folder, test_folder]
    set_names = ["Train", "Validation", "Test"]
    counts = []

    # Count images in each set
    for folder in set_folders:
        total_images = sum([len(os.listdir(os.path.join(folder, label))) for label in os.listdir(folder)])
        counts.append(total_images)

    # Plot the results
    plt.bar(set_names, counts, color=['blue', 'orange', 'green'])
    plt.xlabel("Dataset Type")
    plt.ylabel("Number of Images")
    plt.title("Number of Images in Train, Validation, and Test Sets")
    plt.show()

# Run the visualization
visualize_data_distribution()
