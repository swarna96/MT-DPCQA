import argparse

import numpy as np
import os
import cv2
import open3d as o3d


def background_crop(img):
    """
    Remove white (255) background borders from an image.
    """
    gray_imge = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    column = np.mean(gray_imge, axis=0)
    row = np.mean(gray_imge, axis=1)
    col_left = 0
    col_right= len(column)
    row_top = 0
    row_bottom = len(row)
    
    for i in range(len(column)):
        if column[i] != 255:
            col_left = i
            break

    
    for i in range(len(column)):
        if column[-i-1] != 255:
            col_right = len(column)-i
            break

    for i in range(len(row)):
        if row[i] != 255:
            row_top = i
            break

    for i in range(len(row)):
        if row[-i-1] != 255:
            row_bottom = len(row)-i
            break

    cropped = img[row_top:row_bottom, col_left:col_right, :]
    return cropped

def generate_two_rgb_views(input_path, output_dir):
    """
    Loads a point cloud or mesh (ply/obj) and captures two RGB views:
      - front (0 degrees)
      - back (180 degrees around Y-axis)
    Stores them as 'view0.png' and 'view1.png' in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the point cloud

    geometry = o3d.io.read_point_cloud(input_path)
    

    # Initialize Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1080, height=1920)
    vis.add_geometry(geometry)
    ctrl = vis.get_view_control()

    # Capture front view (0°)
    vis.poll_events()
    vis.update_renderer()
    img_front = vis.capture_screen_float_buffer(True)
    img_front = (np.asarray(img_front) * 255).astype(np.uint8)
    img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
    img_front = background_crop(img_front)
    cv2.imwrite(os.path.join(output_dir, 'view0.png'), img_front)

    
    rotate_interval = 1  # number of times to call rotate
    interval = 5.82  # interval for 1 degree
    for _ in range(rotate_interval):
        ctrl.rotate(180 * interval, 0)
        vis.poll_events()
        vis.update_renderer()

    # Capture the back view
    img_back = vis.capture_screen_float_buffer(True)
    img_back = (np.asarray(img_back) * 255).astype(np.uint8)
    img_back = cv2.cvtColor(img_back, cv2.COLOR_RGB2BGR)
    img_back = background_crop(img_back)
    cv2.imwrite(os.path.join(output_dir, 'view1.png'), img_back)

    # Cleanup
    vis.destroy_window()
    print(f"Saved: {output_dir}/view0.png, {output_dir}/view1.png")


def process_dynamic_pointclouds(root_input_dir, root_output_dir, sampling_interval=5):
    """
    Processes dynamic point clouds, generating projections for every nth frame (sampling_interval).
    Stores outputs in the structured format:
    /home/wnp23/dynamicPCQA/projections/{sequence_name}/{frame_name}/view0.png
    """
    if not os.path.exists(root_output_dir):
        os.makedirs(root_output_dir)

    # Get all sequences inside the dataset directory
    sequences = sorted(os.listdir(root_input_dir))

    for sequence_name in sequences:
        sequence_path = os.path.join(root_input_dir, sequence_name)

        if not os.path.isdir(sequence_path):
            continue  # Skip if not a directory

        # Get all frames inside the sequence
        frames = sorted(os.listdir(sequence_path))

        for i, frame_file in enumerate(frames):
            if i % sampling_interval != 0:
                continue  # Skip frames that don't match the sampling interval

            frame_path = os.path.join(sequence_path, frame_file)
            if not frame_path.endswith('.ply'):  # Ensure we only process .ply files
                continue

            # Define the structured output path
            frame_name = os.path.splitext(frame_file)[0]  # Remove .ply extension
            output_dir = os.path.join(root_output_dir, sequence_name, frame_name)

            # Generate projections
            generate_two_rgb_views(frame_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate RGB projections of dynamic point clouds.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input dynamic point clouds.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save generated projections.')
    parser.add_argument('--interval', type=int, default=5, help='Sampling interval for frames.')
    args = parser.parse_args()

    process_dynamic_pointclouds(args.input_dir, args.output_dir, args.interval)

if __name__ == "__main__":
    main()
