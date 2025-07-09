import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Source and destination folders
source_folder = r'C:\Users\saiki\OneDrive\Desktop\alzheimer_dataset\Alzheimer_s Dataset'
destination_folder = r'C:\Users\saiki\OneDrive\Desktop\alzheimer_dataset\PCA_Applied_Images'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Function to apply PCA to an image
def apply_pca(image, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(image)
    principal_components = pca.components_
    reconstructed_image = np.dot(pca.transform(image), principal_components)
    reconstructed_image = np.reshape(reconstructed_image, image.shape)
    reconstructed_image = reconstructed_image.astype(np.uint8)
    return reconstructed_image

# Iterate over each image in the source folder and apply PCA
for root, _, files in os.walk(source_folder):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):  # You can add more extensions if needed
            # Load the image
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path, 0)

            # Resize the image if needed
            image = cv2.resize(image, (256, 256))

            # Apply PCA
            n_components = 100  # Specify the number of components to keep
            reconstructed_image = apply_pca(image, n_components)

            # Save the PCA-applied image to the destination folder
            class_label = os.path.basename(os.path.dirname(image_path))
            class_folder_dest = os.path.join(destination_folder, class_label)
            if not os.path.exists(class_folder_dest):
                os.makedirs(class_folder_dest)

            destination_path = os.path.join(class_folder_dest, filename)
            cv2.imwrite(destination_path, reconstructed_image)

# Display random images of each class
class_labels = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
num_random_images = 3  # Number of random images to show from each class

# Create subplots
fig, axes = plt.subplots(len(class_labels), num_random_images, figsize=(10, 10))

for i, class_label in enumerate(class_labels):
    class_folder = os.path.join(destination_folder, class_label)
    class_images = os.listdir(class_folder)
    random_images = np.random.choice(class_images, num_random_images, replace=False)

    for j, image_name in enumerate(random_images):
        image_path = os.path.join(class_folder, image_name)
        image = cv2.imread(image_path, 0)

        # Display the image
        axes[i, j].imshow(image, cmap='gray')
        axes[i, j].set_title(f'{class_label}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()