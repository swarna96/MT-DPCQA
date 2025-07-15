# 🚀 MT-DPCQA: MT-DPCQA: A Multimodal Time-aware Learning Approach for No-Reference Dynamic Point Cloud Quality Assessment

Official implementation of the paper "MT-DPCQA: A Multimodal Time-aware Learning Approach for No-Reference Dynamic Point Cloud Quality Assessment" accepted in ACM MM 2025

##  🧪 Environment ##
Ubuntu 22.04.3 LTS 

Python 3.8.18 

Install pytorch, openCV, open3D

## 📂 Dataset structure

/path/to/dataset/ 

├── Sequence_A/ 

│   ├── Frame_000.ply 

│   ├── Frame_001.ply 

│   └── ... 

├── Sequence_B/ 

│   ├── Frame_000.ply 

│   ├── Frame_001.ply 

│   └── ... 

## 🏋️‍♂️ How to Run the Code 
**Generate projections:** 
python generateProjections.py --input_dir  <path to the ply files> 
 --output_dir  <Path to store the projections> --interval 5 

**Generate Patches:** 

 

python generatePatches.py --input_dir  <path to the ply files> --output_dir  <Path to store the patches> 

 



**Train:**

python train.py  --root <path to your ply files> --projections_dir  <path to projection dir> --labels_file  path to the labels file --patch_dir <path to patches>  --train_csv <path to train csv> --test_csv  <path to test csv> --log_file  <path to log file> --checkpoint_path <path to the pointnet checkpoint> 


