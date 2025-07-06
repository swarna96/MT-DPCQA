import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.optimize import curve_fit

def farthest_point_sampling(points, num_centroids):
    N = points.shape[0]
    centroids = torch.zeros((num_centroids, points.shape[1]), device=points.device)
    distances = torch.full((N,), float('inf'), device=points.device)
    torch.manual_seed(0)
    farthest = torch.randint(0, N, (1,), device=points.device).item()

    for i in range(num_centroids):
        centroids[i] = points[farthest]
        dists = torch.sqrt(torch.sum((points - centroids[i]) ** 2, dim=-1))
        distances = torch.min(distances, dists)
        farthest = torch.argmax(distances)

    return centroids

def knn_clustering(points, centroids, k=512):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points.cpu().numpy())
    _, indices = nbrs.kneighbors(centroids.cpu().numpy())
    indices = torch.from_numpy(indices).to(points.device)
    patches = [points[idx] for idx in indices]

    return patches

def get_processed_patches_rgb(point_cloud_tensor, rgb_data, patch_size, point_size):
    patches = []
    centroids = farthest_point_sampling(point_cloud_tensor, patch_size)
    coords = knn_clustering(point_cloud_tensor, centroids, k=point_size)

    patches_rgb = knn_clustering(rgb_data, centroids, k=point_size)
    for i in range(patch_size):
        patches.append(torch.cat([coords[i], patches_rgb[i]], dim=1))
    return patches, coords

def normalize_point_cloud(point_cloud):
    point_cloud_np = point_cloud.cpu().numpy()
    point_cloud_centered = point_cloud_np - np.mean(point_cloud_np, axis=0)
    scale = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
    point_cloud_normalized = point_cloud_centered / scale
    point_cloud_normalized_mean = np.mean(point_cloud_normalized, axis=0)
    point_cloud_normalized_std = np.std(point_cloud_normalized, axis=0)
    point_cloud_normalized = (point_cloud_normalized - point_cloud_normalized_mean) / point_cloud_normalized_std
    point_cloud_normalized = torch.from_numpy(point_cloud_normalized).to(point_cloud.device)
    return point_cloud_normalized

def collate_fn(batch):
    # Batch contains multiple DPCs
    padded_patches = []
    padded_img1 = []
    padded_img2 = []
    labels = []
    dpc_ids = []

    max_frames = max(item[0].shape[0] for item in batch)
    
    for patches, img1, img2, mos, dpc_id in batch:
        T = patches.shape[0]
        pad_size = max_frames - T
        
        padded_patches.append(
            torch.cat([patches, torch.zeros(pad_size, *patches.shape[1:])], dim=0)
        )
        padded_img1.append(
            torch.cat([img1, torch.zeros(pad_size, *img1.shape[1:])], dim=0)
        )
        padded_img2.append(
            torch.cat([img2, torch.zeros(pad_size, *img2.shape[1:])], dim=0)
        )
        labels.append(mos)
        dpc_ids.append(dpc_id)
        
    return (
        torch.stack(padded_patches),  # [B, 60, 100, 3, 1024]
        torch.stack(padded_img1),     # [B, 60, 3, 224, 224]
        torch.stack(padded_img2),
        torch.stack(labels),
        dpc_ids
    )

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic