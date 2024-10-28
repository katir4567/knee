import os 
import numpy as np
from pydicom import dcmread
import matplotlib.pyplot as plt

def coordinates(points_file):
    """ Read landmark coordinates from a file. """
    line_count = 0
    x = []
    y = []
    with open(points_file) as fp:
        for line in fp:
            if line_count >= 1:  # Skip the header line
                end_point = 151
            if 3 <= line_count < end_point:  # Read only desired lines
                x_temp, y_temp = line.split(" ")
                y_temp = y_temp.replace("\n", "")
                x.append(float(x_temp))
                y.append(float(y_temp))
            line_count += 1
    return x, y

def crop_image_with_margin(image, coords, margin_pixels):
    """ Crop the image based on landmark coordinates with a given margin. """
    if not coords:
        return image  # Return the original image if no coordinates

    # Calculate min and max coordinates for cropping
    x_min = int(np.min(coords[0])) - margin_pixels
    x_max = int(np.max(coords[0])) + margin_pixels
    y_min = int(np.min(coords[1])) - margin_pixels
    y_max = int(np.max(coords[1])) + margin_pixels

    # Ensure coordinates are within the image bounds
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Directories for DICOM files and landmark files
dicom_dir = 'data/dicoms'
landmarks_dir = 'data/landmarks'

# Processing each DICOM file
for dcm in os.listdir(dicom_dir):
    dcm_name = os.path.splitext(dcm)[0]
    dcm_path = os.path.join(dicom_dir, dcm)
    dicom_data = dcmread(dcm_path)

    # Load the corresponding landmarks
    landmark_path = os.path.join(landmarks_dir, f"{dcm_name}.pts")
    Xs, Ys = coordinates(landmark_path)

    ############## Showing the original image ###############
    plt.imshow(dicom_data.pixel_array, cmap=plt.cm.gray)
    plt.title(f"Patient ID: {dicom_data.PatientID}")
    plt.axis('off')  # Hide axis for clarity
    plt.show()

    # Visualizing Bone Landmarks #
    plt.imshow(dicom_data.pixel_array, cmap=plt.cm.gray)
    plt.scatter(Xs[:73], Ys[:73], c='green', marker='.', s=10, label='Landmarks 1')
    plt.scatter(Xs[74:], Ys[74:], c='blue', marker='.', s=10, label='Landmarks 2')
    plt.title(f"Patient ID: {dicom_data.PatientID}")
    plt.axis('off')
    plt.legend()
    plt.show()

    # Cropping with Margin #
    margin_cm = 1  # Margin in centimeters
    pixel_spacing = dicom_data.PixelSpacing  # Pixel spacing in mm
    margin_pixels = int(margin_cm / pixel_spacing[0])  # Convert margin to pixels

    # Crop the image around the landmarks
    cropped_image = crop_image_with_margin(dicom_data.pixel_array, (Xs, Ys), margin_pixels)

    # Save the cropped image as PNG
    output_filename = os.path.join('data', f"{dcm_name}_cropped.png")
    plt.imsave(output_filename, cropped_image, cmap='gray')

    # Show the cropped image
    plt.imshow(cropped_image, cmap=plt.cm.gray)
    plt.title(f"Cropped Image: {dcm_name}")
    plt.axis('off')
    plt.show()

    print(f"Cropped image saved as: {output_filename}")
