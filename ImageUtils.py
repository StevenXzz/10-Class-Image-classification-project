import numpy as np
from matplotlib import pyplot as plt
import torchvision

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    # Convert the record from [3072,] to [3, 32, 32]
    image = record.reshape((3, 32, 32))
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    # Normalize the image
    image = image / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.247, 0.243, 0.261]).reshape((3, 1, 1))
    image = (image - mean) / std

    if training:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)

        # Random rotation
        angle = np.random.uniform(-15, 15)  # Rotate by -15 to +15 degrees
        image = rotate_image(image, angle)

        # Random crop with padding
        image = random_crop(image, padding=4)

    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image = image.reshape((3, 32, 32))

    image = np.transpose(image, (1, 2, 0))
    
    plt.imshow(image)
    plt.savefig(save_name)
    plt.show()

    return image   

    ### YOUR CODE HERE

# Other functions
### YOUR CODE HERE
def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    from scipy.ndimage import rotate
    rotated_images = np.zeros_like(image)
    for c in range(image.shape[0]):  # Loop through color channels
        rotated_images[c] = rotate(image[c], angle, reshape=False, mode='nearest')
    return rotated_images

def random_crop(image, padding=4):
    """Apply padding and randomly crop the image."""
    from scipy.ndimage import zoom
    image_padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    crop_start_h = np.random.randint(0, 2 * padding)
    crop_start_w = np.random.randint(0, 2 * padding)
    image_cropped = image_padded[:, crop_start_h:crop_start_h+32, crop_start_w:crop_start_w+32]
    return image_cropped


### END CODE HERE

# Other functions can be added here for additional preprocessing or data manipulation.
