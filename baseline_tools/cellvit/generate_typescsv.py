"""
Lucas Sancéré 2025
"""

import os
import csv

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path


def create_types_csv(folder_path, output_csv):
    # Get all files in the folder (ignore directories)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Open CSV for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["img", "type"])
        
        # Write each file and its type
        for file_name in files:
            writer.writerow([file_name, "Skin"])

    print(f"CSV created at {output_csv} with {len(files)} entries.")





@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    baselinedata_dir= cfg.baselinedata_dir

    folder1_path = baselinedata_dir + '/fold0/images/'
    folder2_path = baselinedata_dir + '/fold1/images/'
    folder3_path = baselinedata_dir + '/fold2/images/'
    folders = [folder1_path, folder2_path, folder3_path]
    output_csv1 = baselinedata_dir + '/fold0/types.csv'
    output_csv2 = baselinedata_dir + '/fold1/types.csv'
    output_csv3 = baselinedata_dir + '/fold2/types.csv'
    outputs = [output_csv1, output_csv2, output_csv3]

    for k in range(0,3):
        create_types_csv(folders[k], outputs[k])



if __name__ == "__main__":
    main()
