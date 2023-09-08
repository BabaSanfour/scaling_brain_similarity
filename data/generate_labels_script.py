import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_data_config_parser


data_paths = {"imagenet": 'ILSVRC/Data/CLS-LOC/train/',
                 "places365": 'places365_standard/train/'}

def assign_labels_to_dirs(folder_path: str) -> dict:
    """
    Assign labels to subdirectories in a folder starting from 0.

    Args:
        folder_path (str): Path to the folder containing subdirectories.
    Returns:
        dict: A dictionary mapping subdirectory names to labels.
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise ValueError("Invalid folder path provided.")

    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    labels = {}
    for idx, subdir in enumerate(subdirs):
        labels[subdir] = idx

    return labels

if __name__ == "__main__":
    parser = get_data_config_parser()
    args = parser.parse_args()

    folder_path = data_paths[args.dataset]
    
    labels = assign_labels_to_dirs(folder_path)

    # Save labels to a JSON file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"match_labels_{args.dataset}.json")
    with open(output_file, "w") as json_file:
        json.dump(labels, json_file, indent=4)

    print(f"Labels saved to {output_file}")