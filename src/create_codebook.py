
# imports for creating codebook
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from torch_cluster import knn

import numpy as np
from matplotlib.pyplot import plot as plt


from torch.autograd import Variable
from tqdm import tqdm

def get_n_batches(data_loader, num_batches):
    collected_batches = 0
    all_samples = []
    all_labels = []
    for samples, labels in data_loader:
        all_samples.append(samples)
        all_labels.append(labels)
        collected_batches += 1
        if collected_batches == num_batches:
            break
    all_samples_tensor = torch.cat(all_samples, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    return all_samples_tensor, all_labels_tensor


def add_figure_encoder(flatten_ze, cluster_centers_):
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    reduced_data_encoder = pca.fit_transform(flatten_ze)
    reduced_data_codebook = pca.fit_transform(cluster_centers_)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'cyan', 'yellow']
    color_index = int(np.log2(len(reduced_data_codebook)))
    x = np.array([elem[0] for elem in reduced_data_encoder])
    y = np.array([elem[1] for elem in reduced_data_encoder])
    name = str(np.log2(len(reduced_data_codebook)) + 'Bits Vectors')
    train_x_vals = np.array([elem[0] for elem in reduced_data_codebook])
    train_y_vals = np.array([elem[1] for elem in reduced_data_codebook])
    plt.scatter(train_x_vals, train_y_vals, s=10, alpha=0.1, label='Train Vectors')
    plt.scatter(x, y, s=250, alpha=1, label=name, c=colors[color_index])

    # Add axis labels and a title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    plt.grid()
    plt.legend(loc='best')
    # Show the plot
    plt.show()
    codebook_size = len(cluster_centers_)
    plt.savefig(f'scatter_plot_codebook_{codebook_size}.png', dpi=300, bbox_inches='tight')


def create_codebook_command(encoder : nn.Sequential, input_dataset, cb_vec_dim, num_clusters):
    print(f'Perofrm Codebook generation for VQ-VAE architecture')

    # K-Means Quantization Setup
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 11,
        "max_iter": 100,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flatten_ze_sub = []
    for data in tqdm(input_dataset):
        Rx, DOA = data

        # Cast observations and DoA to Variables
        Rx = Rx.to(device)

        with torch.no_grad():
            z_e = encoder(Rx)
        z_e = z_e - z_e.mean()
        if torch.is_complex(z_e):
            z_e = torch.view_as_real(z_e)

        flatten_ze_sub.append(z_e.view(-1, cb_vec_dim))

    #kmeans = KMeans(num_clusters, **kmeans_kwargs)
    flatten_ze = torch.cat(flatten_ze_sub, dim=0)
    #kmeans.fit(flatten_ze.cpu().numpy())
    #flatten_ze = torch.cat(flatten_ze_sub, dim=0)
    #codebook_vectors = torch.Tensor(kmeans.cluster_centers_)
    #codebook_vectors = get_codebook_vectors(flatten_ze, num_clusters, num_iters=50)
    codebook_vectors = get_codebook_vectors_low_profile(flatten_ze, num_clusters, num_iters=50)
    
    #add_figure_encoder(z_e, kmeans.cluster_centers_)
    return codebook_vectors


def batch_cdist_and_argmin(data, centroids, cluster_assignments_temp, batch_size):
    """
    Compute batched pairwise distances (cdist) and global argmin.

    Args:
        data (torch.Tensor): Data points of shape (N, D).
        centroids (torch.Tensor): Centroids of shape (K, D).
        batch_size (int): Batch size for processing.

    Returns:
        torch.Tensor: Cluster assignments for each data point, shape (N,).
    """
    device = data.device
    num_points = data.size(0)
    num_centroids = centroids.size(0)



    # Compute distances and assignments batch-wise
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch = data[start:end]  # Get the current batch

        # Compute distances for the batch
        if torch.is_complex(batch):
            batch_real = torch.view_as_real(batch)
            batch = batch_real
        distances = torch.cdist(batch, centroids, p=2)  # Shape: (batch_size, num_centroids)

        # Find the index of the closest centroid (argmin)
        cluster_assignments_temp[start:end] = torch.argmin(distances, dim=1)

    return cluster_assignments_temp



def get_codebook_vectors(flatten_ze, num_clusters, num_iters):

    """
    Perform k-means clustering using PyTorch on GPU.

    Args:
        flatten_ze (torch.Tensor): Data to cluster, shape (N, D). Should be on GPU.
        num_clusters (int): Number of clusters.
        num_iters (int): Number of k-means iterations.

    Returns:
        torch.Tensor: Codebook vectors (cluster centers), shape (num_clusters, D).
    """
    #assert flatten_ze.is_cuda, "Input data must be on GPU"
    assert flatten_ze.ndim == 2, "Input tensor must be 2D (N, D)"

    device = flatten_ze.device
    N, D = flatten_ze.shape

    # Randomly initialize cluster centers
    indices = torch.randperm(N)[:num_clusters]
    centroids = flatten_ze[indices]  # Initial cluster centers, shape (num_clusters, D)

    # Initialize tensor for cluster assignments
    cluster_assignments_temp = torch.empty(N, dtype=torch.long, device=device)

    for index in range(num_iters):
        """
        Calculate the batch distance and then select centeriod globally
        """
        cluster_assignments = batch_cdist_and_argmin(flatten_ze, centroids, cluster_assignments_temp, batch_size=1024)
        # Compute distances and assign each point to the nearest centroid
        #distances = torch.cdist(flatten_ze, centroids, p=2)  # Shape: (N, num_clusters)
        #cluster_assignments = torch.argmin(distances, dim=1)  # Shape: (N,)
        print(f'Calculate Cenetroids for LBG iteration {index + 1} / {num_iters}')
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            members = flatten_ze[cluster_assignments == k]
            if members.size(0) > 0:
                new_centroids[k] = members.mean(dim=0)

        # Check for convergence (optional)
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids

def get_codebook_vectors_low_profile(flatten_ze, num_clusters, num_iters):
    """
    Perform k-means clustering using PyTorch on GPU.

    Args:
        flatten_ze (torch.Tensor): Data to cluster, shape (N, D). Should be on GPU.
        num_clusters (int): Number of clusters.
        num_iters (int): Number of k-means iterations.

    Returns:
        torch.Tensor: Codebook vectors (cluster centers), shape (num_clusters, D).
    """
    #assert flatten_ze.is_cuda, "Input data must be on GPU"
    assert flatten_ze.ndim == 2, "Input tensor must be 2D (N, D)"

    device = flatten_ze.device
    N, D = flatten_ze.shape

    # Randomly initialize cluster centers
    indices = torch.randperm(N)[:num_clusters]
    centroids = flatten_ze[indices]  # Initial cluster centers, shape (num_clusters, D)

    # Initialize tensor for cluster assignments
    cluster_assignments_temp = torch.empty(N, dtype=torch.long, device=device)
    unit_cluster_assignments = torch.ones(N, dtype=torch.float, device=device)
    for index in range(num_iters):
        """
        Calculate the batch distance and then select centroid globally
        """
        cluster_assignments = batch_cdist_and_argmin(flatten_ze, centroids, cluster_assignments_temp, batch_size=1024)

        print(f'Calculate Centroids for LBG iteration {index + 1} / {num_iters}')

        # Initialize centroid sums and counts
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, device=flatten_ze.device)

        # Incrementally calculate the mean for each cluster
        new_centroids.scatter_add_(0, cluster_assignments.unsqueeze(1).expand(-1, flatten_ze.size(1)), flatten_ze)
        counts.scatter_add_(0, cluster_assignments, unit_cluster_assignments)

        # Finalize the new centroids by dividing by counts
        valid_clusters = counts > 0
        new_centroids[valid_clusters] /= counts[valid_clusters].unsqueeze(1)

        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids


def init_weights_lbg(module, codebook):
    weight_tensor = torch.Tensor(codebook)
    module.weight = nn.Parameter(weight_tensor)
    return module.weight


def get_min_max(encoder : nn.Sequential, input_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_max_ze = float('-inf')
    global_min_ze = float('inf')

    for data in tqdm(input_dataset):
        Rx, DOA = data

        # Cast observations and DoA to Variables
        Rx = Rx.to(device)

        with torch.no_grad():
            z_e = encoder(Rx)

            if torch.is_complex(z_e):
                z_e = torch.view_as_real(z_e)

            global_max_ze = max(global_max_ze, torch.max(z_e))  # Update max
            global_min_ze = min(global_min_ze, torch.min(z_e))  # Update min

    return global_max_ze, global_min_ze