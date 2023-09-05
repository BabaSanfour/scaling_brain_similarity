import os
import time
import shutil
import random
import logging

# Configure logging
logging.basicConfig(filename='data_copy.log', level=logging.INFO)

def get_all_files(directory: str) -> list:
    """
    Get a list of all files in the specified directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of file names in the directory.
    """
    filenames = []
    class_name = os.path.basename(directory)
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
            part_name = filename.split('_')[0]
            assert part_name == class_name
    return filenames

def copy_data(scaling_factor: int, dirs: list, subtrain_per_class: str) -> None:
    """
    Copy data from source directories to subdirectories with a specified scaling factor.

    Args:
        scaling_factor (int): Scaling factor for the number of images.
        dirs (list): List of class directories.
        subtrain_per_class (str): Path to the target directory.
    """
    process_start_time = time.time()
    total_images_per_scaling_factor = 0

    for directory in dirs:
        directory_path = os.path.join(train_path, directory)
        filenames = get_all_files(directory_path)
        original_number_images_per_class = len(filenames)
        new_number_images_per_class = max(int(original_number_images_per_class / (2 ** scaling_factor)), 1)
        
        # Randomly sample if there are more images than needed
        if len(filenames) > new_number_images_per_class:
            filenames = random.sample(filenames, new_number_images_per_class)

        total_images_per_scaling_factor += len(filenames)
        
        # Create the target directory if it doesn't exist
        os.makedirs(os.path.join(subtrain_per_class, directory), exist_ok=True)

        for filename in filenames:
            src_file_path = os.path.join(train_path, directory, filename)
            tgt_file_path = os.path.join(subtrain_per_class, directory, filename)

            # Use shutil for file copying
            try:
                shutil.copy(src_file_path, tgt_file_path)
            except Exception as e:
                logging.error(f"Error copying file {filename}: {str(e)}")
    
    logging.info(f'Finished {scaling_factor} scaling factor, total cost time: {time.time() - process_start_time:.2f} sec')
    logging.info(f'Total number of images selected: {total_images_per_scaling_factor}')

if __name__ == '__main__':
    train_path = 'ILSVRC/Data/CLS-LOC/train/'
    dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    logging.info('Number of classes: %d', len(dirs))
    
    new_train_path = 'imagenet_subtrain/'
    os.system(f'rm -rf {new_train_path}')
    os.system(f'mkdir {new_train_path}')
    
    scaling_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Scaling factor of 0 => get the total dataset
    
    for scaling_factor in scaling_factors:
        subtrain_per_class = os.path.join(new_train_path, 'scaling_fac_' + str(scaling_factor))
        os.system(f'rm -rf {subtrain_per_class}')
        os.system(f'mkdir {subtrain_per_class}')
        copy_data(scaling_factor, dirs, subtrain_per_class)

    logging.info('Finished copying data')