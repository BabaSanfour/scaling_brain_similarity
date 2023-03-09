import os
import time
import random

train_path = 'ILSVRC/Data/CLS-LOC/train/'

def get_all_files(directory):
    filenames = []
    class_name = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
            part_name = filename.split('_')[0]
            assert part_name == class_name
    return filenames

def copy_data(scaling_factor, dirs, subtrain_per_class):
    process_start_t = time.time()
    total_images_per_scaling_factor = 0
    for directory in dirs:
        directory_path = os.path.join(train_path, directory)
        filenames = get_all_files(directory_path)
        original_number_images_per_class = len(filenames)
        new_number_images_per_class = original_number_images_per_class*2^(-scaling_factor)
        if len(filenames)>new_number_images_per_class:
            filenames = random.sample(filenames, new_number_images_per_class)
        total_images_per_scaling_factor += len(filenames)
        os.system(f"mkdir {os.path.join(subtrain_per_class, directory)}")
        
        for filename in filenames:
            src_file_path = os.path.join(train_path, directory, filename)
            tgt_file_path = os.path.join(subtrain_per_class,  directory, filename)
            os.system(f'cp {src_file_path} {tgt_file_path}')
                    
    print(f'finished {scaling_factor} scaling factor,\
            total cost time: {time.time()-process_start_t:.2f} sec')
    print(f'Total number of images selected: {total_images_per_scaling_factor}')

if __name__ == '__main__':
    
    dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print('Number of classes: ', len(dirs))
    new_train_path = 'imagenet_subtrain/'
    os.system(f'rm -rf {new_train_path}')
    os.system(f'mkdir {new_train_path}')
    dataset_length = len(dirs)
    scaling_factors = [1,2,3,4,5,6] # scaling factor of 0 => get the total dataset
    for scaling_factor in scaling_factors:
        subtrain_per_class = os.path.join(new_train_path, 'scaling_fac_' + str(scaling_factor))
        os.system(f'rm -rf {subtrain_per_class}')
        os.system(f'mkdir {subtrain_per_class}')
        copy_data(scaling_factor, dirs, subtrain_per_class)


