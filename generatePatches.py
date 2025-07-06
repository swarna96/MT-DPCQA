import argparse

from utils import normalize_point_cloud, get_processed_patches_rgb
import os
import open3d as o3d
import pickle
import numpy as np
import torch
def prepare_frame_list(root_dir, cache_dir):
    frame_list = []
    for dpc_folder in sorted(os.listdir(root_dir)):
        dpc_folder_path = os.path.join(root_dir, dpc_folder)
        if os.path.isdir(dpc_folder_path):
            dpc_cache_dir = os.path.join(cache_dir, dpc_folder)
            if not os.path.exists(dpc_cache_dir):
                os.makedirs(dpc_cache_dir)

            #label = self.labels[dpc_folder]

            # Sort the files and take every 5th one
            sorted_files = sorted(os.listdir(dpc_folder_path))
            sampled_files = sorted_files[::5]  # Take every 5th frame

            for idx, file_name in enumerate(sampled_files):
                file_name_without_ext = os.path.splitext(file_name)[0]
                frame_cache_file = os.path.join(dpc_cache_dir, f"{file_name_without_ext}.pkl")

                if not os.path.exists(frame_cache_file):
                    save_frame_patches(dpc_folder_path, file_name, dpc_folder, idx, frame_cache_file)

                frame_list.append((frame_cache_file, dpc_folder, file_name_without_ext))

    return frame_list


def save_frame_patches(dpc_folder_path, file_name, dpc_folder, frame_idx, frame_cache_file):
    file_path = os.path.join(dpc_folder_path, file_name)
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if points.size == 0:
        print(f"Skipping empty point cloud: {file_path}")
        return

    points = normalize_point_cloud(torch.FloatTensor(points).to("cpu"))
    colors = torch.FloatTensor(colors).to("cpu")

    features, coords = get_processed_patches_rgb(points, colors, 100, 1024)
    patches = [{'feature': coord} for coord in coords]

    with open(frame_cache_file, 'wb') as f:
        pickle.dump(patches, f)


def main():
    parser = argparse.ArgumentParser("Generate patches of dynamic point clouds.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input dynamic point clouds.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save generated patches.')
    args = parser.parse_args()

    prepare_frame_list(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()