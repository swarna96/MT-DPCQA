import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms

class Fusion_Dataset(Dataset):
    def __init__(self, root_dir, projections_dir, labels_file, patch_dir, csv_file):
        self.root_dir = root_dir
        self.patch_dir = patch_dir
        self.projections_dir = projections_dir
        self.labels = self.load_labels(labels_file)
        self.patch_size = 100
        self.point_size = 1024

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.sequence_list = self.load_csv_sequences(csv_file)

        self.dpc_list = self.prepare_dpc_list()

    def load_csv_sequences(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        return df.iloc[:, 0].tolist()

    def prepare_dpc_list(self):
        dpc_list = []
        for dpc_folder in sorted(self.sequence_list):
            dpc_path = os.path.join(self.root_dir, dpc_folder)
            if os.path.isdir(dpc_path):
                all_files = sorted(os.listdir(dpc_path))
                assert len(all_files[::5]) == 60, f"Invalid frame count for {dpc_folder}"
                dpc_list.append({
                    'dpc_id': dpc_folder,
                    'label': self.labels[dpc_folder],
                    'frame_files': [f for f in all_files[::5]]
                })
        return dpc_list

    def __len__(self):
        return len(self.dpc_list)
    
    def load_labels(self, labels_file):
        if labels_file.endswith('.csv'):
            df = pd.read_csv(labels_file)
            labels = dict(zip(df['Point_Cloud_Name'], df['MOS']))
        else:
            df = pd.read_excel(labels_file, header=None)
            labels = dict(zip(df.iloc[:, 1], df.iloc[:, 2]))
        return labels

    def __getitem__(self, idx):
        dpc_info = self.dpc_list[idx]
        dpc_id = dpc_info['dpc_id']
        mos = dpc_info['label']

        all_patches = []
        all_img1 = []
        all_img2 = []

        for frame_file in dpc_info['frame_files']:
            frame_name = os.path.splitext(frame_file)[0]

            frame_cache_file = os.path.join(self.patch_dir, dpc_id, f"{frame_name}.pkl")

            if not os.path.exists(frame_cache_file):
                print(f"Warning: Missing patch file {frame_cache_file}, skipping this frame.")
                continue

            with open(frame_cache_file, 'rb') as f:
                patches = pickle.load(f)


            selected_indices = torch.randperm(len(patches))[:self.patch_size]
            selected_patches = [patches[i] for i in selected_indices]
            feature_list = [torch.tensor(patch['feature'].T, dtype=torch.float32) for patch in selected_patches]
            all_patches.append(torch.stack(feature_list))

            view0_path = os.path.join(self.projections_dir, dpc_id, frame_name, "view0.png")
            view1_path = os.path.join(self.projections_dir, dpc_id, frame_name, "view1.png")

            img1 = self.image_transform(Image.open(view0_path).convert("RGB"))
            img2 = self.image_transform(Image.open(view1_path).convert("RGB"))

            all_img1.append(img1)
            all_img2.append(img2)

        return (
            torch.stack(all_patches),  # [60, 100, 3, 1024]
            torch.stack(all_img1),     # [60, 3, 224, 224]
            torch.stack(all_img2),     # [60, 3, 224, 224]
            torch.tensor(mos, dtype=torch.float32),
            dpc_id
        )
