import torch
import torch.nn as nn
import torch.nn.functional as F
# FedAnil+: K-Medoids (Custom implementation to avoid sklearn_extra numpy incompatibility)
import copy
from sklearn.metrics import pairwise_distances

class KMedoids:
    """Lightweight KMedoids drop-in replacement for sklearn_extra.cluster.KMedoids"""
    def __init__(self, n_clusters=2, random_state=0, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None
        self.medoid_indices_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        X = np.array(X)
        n_samples = X.shape[0]
        D = pairwise_distances(X)
        # Initialize medoids randomly
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        for _ in range(self.max_iter):
            # Assign labels
            labels = np.argmin(D[:, indices], axis=1)
            new_indices = np.copy(indices)
            for k in range(self.n_clusters):
                cluster_mask = np.where(labels == k)[0]
                if len(cluster_mask) == 0:
                    continue
                # Pick the point with minimum total distance to others in cluster
                sub_D = D[np.ix_(cluster_mask, cluster_mask)]
                new_indices[k] = cluster_mask[np.argmin(sub_D.sum(axis=1))]
            if np.array_equal(new_indices, indices):
                break
            indices = new_indices
        self.medoid_indices_ = indices
        self.labels_ = np.argmin(D[:, indices], axis=1)
        self.cluster_centers_ = X[indices]
        return self
# FedAnil+: Silhouette Index
from sklearn.metrics import silhouette_score
import numpy as np

# Define ResNet50 model
#resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

# Define the ResNet50 architecture using nn.Sequential
resnet50 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2), 
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Linear(7*7*64, 512),
    nn.Linear(512, 10),)

# Define GloVe model
glove = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
)

# Define CNN model
cnn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(3136, 512),
    nn.Linear(512, 10),
    nn.ReLU(),
)

# Define concatenated model
class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50
        self.glove = glove
        self.cnn = cnn
        self.fc3 = nn.Linear(1000 + 16 + 128, 256) # Concatenated output size is 1000+16+128 = 1144
        self.fc4 = nn.Linear(256, 10) # Output size is 10 for classification
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        resnet_outetput = tensor = F.relu(self.bn1(self.conv1(tensor)))
        tensor = self.pool1(tensor)
        glove_output = tensor = F.relu(self.bn2(self.conv2(tensor)))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        cnn_output = tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        #concat = torch.cat((resnet_outetput, glove_output, cnn_output), dim=1)
        #x = F.relu(self.fc1(concat))
        return tensor
    
    # FedAnil+: Sparsification
    def first_filter(self, global_parameters):
        selected_parameters = {}
        for var in self.state_dict():
            shape_of_original_gradients = self.state_dict()[var].shape 
            reshape_of_local_gradients = self.state_dict()[var].view(-1)
            reshape_of_global_gradients = global_parameters[var].view(-1)
            combine_gradients = reshape_of_global_gradients
            index = 0
            for item1, item2 in zip(reshape_of_local_gradients, reshape_of_global_gradients):
                if item1 > item2:
                    combine_gradients[index] = item1
                else:
                    combine_gradients[index] = 0
                index += 1
            selected_parameters[var] = combine_gradients.reshape(shape_of_original_gradients)
        return selected_parameters
    # FedAnil+: K-Medoids
    def kmedoids_update(self, max_k = 2):
        # FedAnil+: Optimized Clustering (Simulation Mode)
        # Using fixed K=1 for near-instant execution in high-memory simulation
        best_kmedoids_data = {}
        for var in self.state_dict():
            shape_of_datas = self.state_dict()[var].shape
            if len(shape_of_datas) < 2 or shape_of_datas[0] < 3 or self.state_dict()[var].numel() > 10000:
                continue
            # Optimized: Keep on CPU numpy for clustering math
            datas = self.state_dict()[var].reshape(shape_of_datas[0], -1).detach().cpu().numpy()
            cur_k = min(max_k, shape_of_datas[0] - 1)
            if cur_k < 2: continue
            datakm = KMedoids(n_clusters=cur_k, random_state=0).fit(datas)
            best_kmedoids_data[var] = datakm
            del datas
        
        import gc
        gc.collect()
        return best_kmedoids_data
class CombinedModel(nn.Module):
    def __init__(self, glove_model = glove, resnet_model = resnet50, cnn_model = cnn):
        super().__init__()
        self.glove_model = glove_model
        self.resnet_model = resnet_model
        self.cnn_model = cnn_model

    def forward(self, x, model_choice = "cnn"):
        #print(f"X input size {x.size()}")
        x = x.view(-1, 1, 28, 28)
        #print(f"X view input size {x.size()}")
        if model_choice == "glove":
            x = self.glove_model(x)
        elif model_choice == "resnet":
            x = self.resnet_model(x)
        elif model_choice == "cnn":
            for layer in self.cnn_model:
                x = layer(x)
                #print(f"layers {x.size()}")
        else:
            raise ValueError("Invalid model choice.")
        return x
    # FedAnil+: Sparsification
    def first_filter(self, global_parameters):
        selected_parameters = {}
        for var in self.state_dict():
            shape_of_original_gradients = self.state_dict()[var].shape 
            reshape_of_local_gradients = self.state_dict()[var].view(-1)
            reshape_of_global_gradients = global_parameters[var].view(-1)
            combine_gradients = reshape_of_global_gradients
            index = 0
            for item1, item2 in zip(reshape_of_local_gradients, reshape_of_global_gradients):
                if item1 > item2:
                    combine_gradients[index] = item1
                else:
                    combine_gradients[index] = 0
                index += 1
            selected_parameters[var] = combine_gradients.reshape(shape_of_original_gradients)
        return selected_parameters
    # FedAnil+: K-Medoids
    def kmedoids_update(self, max_k = 2):
        # FedAnil+: Optimized Clustering (Simulation Mode)
        # Using fixed K=1 for near-instant execution in high-memory simulation
        best_kmedoids_data = {}
        for var in self.state_dict():
            shape_of_datas = self.state_dict()[var].shape
            if len(shape_of_datas) < 2 or shape_of_datas[0] < 3 or self.state_dict()[var].numel() > 10000:
                continue
            # Optimized: Keep on CPU numpy for clustering math
            datas = self.state_dict()[var].reshape(shape_of_datas[0], -1).detach().cpu().numpy()
            cur_k = min(max_k, shape_of_datas[0] - 1)
            if cur_k < 2: continue
            datakm = KMedoids(n_clusters=cur_k, random_state=0).fit(datas)
            best_kmedoids_data[var] = datakm
            del datas
            
        import gc
        gc.collect()
        return best_kmedoids_data
    @staticmethod
    def compute_similarity_matrix(local_params_list, global_params, dev):
        """
        Turbo Mode: Computes similarity matrix using vectorized operations on GPU.
        Slashing processing time by removing O(N_Clients * N_Layers) Python loops.
        RAM Fix: Wrapped in torch.no_grad() to prevent computation graph accumulation.
        """
        import gc
        num_clients = len(local_params_list)
        if num_clients == 0:
            return np.array([[]])
            
        vars_list = [v for v in global_params.keys() if 'weight' in v or 'bias' in v]
        num_vars = len(vars_list)
        sim_matrix = np.zeros((num_clients, num_vars))
        
        # RAM FIX: no_grad prevents PyTorch from storing computation graphs
        with torch.no_grad():
            # Flatten global params once and keep on device
            global_flat = {var: global_params[var].detach().view(-1).to(dev) for var in vars_list}
                
            for j, var in enumerate(vars_list):
                try:
                    # Optimized: Stack all client parameters for this specific layer/variable
                    # and compute cosine similarity in a single GPU operation
                    all_l_v = torch.stack([lp[var].detach().view(-1).to(dev) for lp in local_params_list])
                    g_v = global_flat[var].unsqueeze(0) # (1, D)
                    
                    # Compute cosine similarity across all clients at once
                    sims = torch.nn.functional.cosine_similarity(all_l_v, g_v, dim=1)
                    sim_matrix[:, j] = sims.cpu().numpy()
                    
                    # Cleanup to keep memory low
                    del all_l_v, g_v, sims
                except Exception as e:
                    print(f"Vectorized similarity failed for layer {var}: {e}. Falling back to zero.")
                    sim_matrix[:, j] = 0.0
            
            # Final cleanup - release all GPU references
            del global_flat
        
        # Force garbage collection to free leaked tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        return sim_matrix

    @staticmethod
    def aggregate_best_cluster(local_params_list, labels, best_cluster_idx):
        """
        Optimized for 12GB RAM: Aggregates parameters in-place without deep copies.
        """
        selected_indices = np.where(labels == best_cluster_idx)[0]
        num_participants = len(selected_indices)
        
        if num_participants == 0:
            return None
            
        sum_params = {}
        for i, idx in enumerate(selected_indices):
            params = local_params_list[idx]
            if i == 0:
                # Initialize sum_params with the first model's structure
                for var, tensor in params.items():
                    sum_params[var] = tensor.detach().clone()
            else:
                for var, tensor in params.items():
                    sum_params[var] += tensor
            
        # Average the parameters in-place
        for var in sum_params:
            sum_params[var] = sum_params[var] / num_participants
            
        return sum_params

class Generator(nn.Module):
    def __init__(self, model='cnn'):
        super().__init__()
        mm = None
        if model == 'resnet':
            self.fc = nn.Linear(10, 512)
            self.fc2 = nn.Linear(512, 7764)
            mm = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.Sigmoid()
            )
        elif model == 'glove':
            mm = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
            )
        elif model == 'cnn':
            mm = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
            )
        self.model = mm
    
    def forward(self, x, model_type = "cnn"):
        output = None
        if model_type == "resnet":
            x = self.fc(x)
            x = self.fc2(x)
            x = x.view(-1, 64, 7, 7)  # Reshape into feature maps
            output = self.model(x)
        else:
            output = self.model(x)
        return output
