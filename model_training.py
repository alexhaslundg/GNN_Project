import torch
import pandas as pd
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
# import the standard scaler from sklearn
from sklearn.preprocessing import StandardScaler
# import scalers from sklearn
from sklearn.preprocessing import StandardScaler
# import the mean_squared_error from sklearn
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Import required libraries
from dscribe.descriptors import CoulombMatrix


def gen_CM(atoms, reshape = False):
    atomic_numbers = len(atoms.get_atomic_numbers())
    coulomb = CoulombMatrix(n_atoms_max=atomic_numbers)
    coulomb_descriptor = coulomb.create(atoms)
    if reshape:
        return coulomb_descriptor.reshape(atomic_numbers, atomic_numbers)
    return coulomb_descriptor

    

def load_model(model_path='best_model.pt', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = GraphNet(node_dim=1, edge_dim=1, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


# Function for inference on a single molecule
def predict_properties(model, CM, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert Coulomb matrix to graph
    graph_data = convert_matrix_to_graph(CM)
    
    # Move data to device
    graph_data = graph_data.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(graph_data)
    
    # Extract results
    results = {
        'energy': outputs['energy'].item(),
        'KS_gap': outputs['KS_gap'].item(),
        'E_gap': outputs['E_gap'].item()
    }
    
    return results

# Function for batch inference on multiple molecules
# Function for batch inference on multiple molecules with denormalization
def predict_batch_properties(model, df, scalers=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Normalize input data if scalers are provided
    if scalers is not None:
        for target in ['energy', 'KS_gap', 'E_gap']:
            if target in df_copy.columns and target in scalers:
                df_copy.loc[~df_copy[target].isna(), target] = scalers[target].transform(
                    df_copy.loc[~df_copy[target].isna(), target].values.reshape(-1, 1)
                ).flatten()
    
    # Prepare test data
    test_data = prepare_dataset(df_copy)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
    # Evaluate
    model.eval()
    predictions = {'energy': [], 'KS_gap': [], 'E_gap': []}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            # Store predictions (still normalized)
            for key in outputs:
                predictions[key].extend(outputs[key].cpu().numpy().flatten())
    
    # Denormalize predictions if scalers are provided
    if scalers is not None:
        for key in predictions:
            if key in scalers and len(predictions[key]) > 0:
                pred_array = np.array(predictions[key]).reshape(-1, 1)
                predictions[key] = scalers[key].inverse_transform(pred_array).flatten()
    
    # Create a DataFrame with predictions (denormalized)
    results_df = pd.DataFrame({
        'energy_pred': predictions['energy'],
        'KS_gap_pred': predictions['KS_gap'],
        'E_gap_pred': predictions['E_gap']
    })
    
    # Add true values if available (from original df, not normalized)
    for key in ['energy', 'KS_gap', 'E_gap']:
        if key in df.columns:
            results_df[f'{key}_true'] = df[key].values
    
    # Add physics validation check
    results_df['physics_valid'] = results_df['KS_gap_pred'] <= results_df['E_gap_pred']
    results_df['physics_violation'] = np.maximum(0, results_df['KS_gap_pred'] - results_df['E_gap_pred'])
    
    return results_df

def comprehensive_evaluation(y_true, y_pred, property_name="Property"):
    """Evaluate predictions with multiple metrics"""

    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Normalized metrics
    data_range = max(y_true) - min(y_true)
    mean_value = np.mean(y_true)
    nmse = mse / np.var(y_true)
    range_percentage_error = mae / data_range * 100
    
    # Mean baseline comparison
    mean_baseline_mse = mean_squared_error(y_true, [mean_value] * len(y_true))
    improvement_over_mean = (1 - mse/mean_baseline_mse) * 100
    
    print(f"=== {property_name} Evaluation ===")
    print(f"Data range: {min(y_true):.4f} to {max(y_true):.4f} (span: {data_range:.4f})")
    print(f"Mean value: {mean_value:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"NMSE (normalized by variance): {nmse:.4f}")
    print(f"Error as % of data range: {range_percentage_error:.2f}%")
    print(f"Improvement over mean baseline: {improvement_over_mean:.2f}%")
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{property_name}: Predicted vs Actual (R²={r2:.4f})')
    ax1.grid(True)
    
    # Residuals vs Actual
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{property_name}: Residuals vs Actual')
    ax2.grid(True)
    
    # Distribution of predictions and actual values
    ax3.hist(y_true, bins=15, alpha=0.5, label='True')
    ax3.hist(y_pred, bins=15, alpha=0.5, label='Predicted')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{property_name}: Distribution Comparison')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'nmse': nmse,
        'range_percentage_error': range_percentage_error,
        'improvement_over_mean': improvement_over_mean
    }
    
    
# Function to convert Coulomb matrix to graph object
def convert_matrix_to_graph(CM, target_values=None):
    n = CM.shape[0]  # Number of nodes (atoms)
    
    # Extract node features from diagonal (atomic properties)
    # For more sophisticated approach, you can use the whole row as node features
    node_features = torch.diag(torch.tensor(CM, dtype=torch.float32)).view(-1, 1)
    
    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []
    
    for i in range(n):
        for j in range(i):
            if i != j:  # Skip self-loops if not needed
                if CM[i, j] != 0:
                        
                    edge_index.append([i, j])
                    edge_attr.append([CM[i, j]])  # Edge weight from Coulomb matrix
    
    if not edge_index:  # Handle single-atom case
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Add target values if provided
    if target_values is not None:
        for key, value in target_values.items():
            data[key] = torch.tensor([value], dtype=torch.float32)
    
    return data


def normalize_data(df):
    """
    Normalizes the Coulomb matrices and target values in the dataframe.
    """
    # Create a copy of the dataframe
    normalized_df = df.copy()
    
    # Option 2: Normalize each Coulomb matrix individually
    # Calculate mean and std across all matrices
    all_values = np.concatenate([cm.flatten() for cm in df['CM']])
    cm_mean = "nan" #np.mean(all_values)
    cm_std = "nan" # np.std(all_values)
    
    # # Normalize each matrix
    # normalized_df['CM'] = normalized_df['CM'].apply(lambda x: (x - cm_mean) / (cm_std + 1e-8))
    
    # Create scalers for each target
    scalers = {}
    for target in ['energy', 'KS_gap', 'E_gap']:
        if target in df.columns:
            # Extract non-NaN values
            target_values = df[target].dropna().values.reshape(-1, 1)
            
            if len(target_values) > 0:
                scaler = StandardScaler()
                scaler.fit(target_values)
                
                # Store scaler for later denormalization
                scalers[target] = scaler
                
                # Apply normalization to all non-NaN values
                normalized_df.loc[~normalized_df[target].isna(), target] = scaler.transform(
                    normalized_df.loc[~normalized_df[target].isna(), target].values.reshape(-1, 1)
                ).flatten()
    
    return normalized_df, scalers, {'cm_mean': cm_mean, 'cm_std': cm_std}

# Function to denormalize predictions
def denormalize_predictions(predictions, scalers):
    """
    Denormalizes predictions back to original scale.
    
    Args:
        predictions: Dictionary of predictions from the model
        scalers: Dictionary of fitted StandardScalers
        
    Returns:
        Dictionary of denormalized predictions
    """
    denormalized = {}
    for key, values in predictions.items():
    
        if key in scalers and values is not None and len(values) > 0:
            values_array = np.array(values).reshape(-1, 1)
            denormalized[key] = scalers[key].inverse_transform(values_array).flatten()
        else:
            denormalized[key] = values
            
    return denormalized


# Graph Neural Network Model for Graph-level Prediction
class GraphNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(GraphNet, self).__init__()
        
        # Node embedding layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Add more expressive power to your model
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        
        self.energy_predictor =  nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),  # Allows outputs between -1 and 1
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.KS_gap_predictor =  nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),  # Allows outputs between -1 and 1
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.E_gap_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),  # Allows outputs between -1 and 1
            nn.Linear(hidden_dim//2, 1)
        )
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, data):
      # Extract attributes from the data object
      x, edge_index, batch = data.x, data.edge_index, data.batch
      
      # If batch is None (during single graph inference), create it
      if batch is None:
          batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
      
      # Initial node embedding
      x = self.node_encoder(x)
    
      # Add residual connections
      x1 = self.conv1(x, edge_index)
      x1 = self.batch_norm1(x1)
      x1 = F.relu(x1)
      
      x2 = self.conv2(x1, edge_index)
      x2 = self.batch_norm2(x2)
      x2 = F.relu(x2) + x1  # Residual connection
      
      x3 = self.conv3(x2, edge_index)
      x3 = self.batch_norm3(x3)
      x3 = F.relu(x3) + x2  # Residual connection
      
      # Graph-level pooling
      x = global_mean_pool(x3, batch)
      
      # Predict multiple properties
      energy = self.energy_predictor(x)
      ks_gap = self.KS_gap_predictor(x)
      e_gap = self.E_gap_predictor(x)

      
      return {
          'energy': energy,
          'KS_gap': ks_gap,
          'E_gap': e_gap
      }

# Prepare dataset
def prepare_dataset(df):
    data_list = []
    
    for idx, row in df.iterrows():
        CM = row['CM']
        
        # Collect all target values available in the dataframe
        target_values = {}
        for target in ['energy', 'KS_gap', 'E_gap']:
            if target in row and not pd.isna(row[target]):
                target_values[target] = row[target]
        
        # Convert matrix to graph object with targets
        graph_data = convert_matrix_to_graph(CM, target_values)
        data_list.append(graph_data)
    
    return data_list

# Training function with multi-task learning
def train(model, loader, optimizer, device, physics_loss_weight=1.0, scalers=None):
    model.train()
    total_loss = 0
    metrics = {'energy_loss': 0, 'KS_gap_loss': 0, 'E_gap_loss': 0, 'count': 0}
    
    # For gradient monitoring
    all_gradients = {}
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss for each target if available
        loss = 0
        energy_weight = 1
        ks_gap_weight = 1
        e_gap_weight = 1
        if hasattr(batch, 'energy'):
            energy_loss = F.mse_loss(outputs['energy'], batch.energy.view(-1, 1))
            loss += energy_loss*energy_weight
            metrics['energy_loss'] += energy_loss.item() * batch.num_graphs
        
        if hasattr(batch, 'KS_gap'):
            ks_gap_loss = F.mse_loss(outputs['KS_gap'], batch.KS_gap.view(-1, 1))
            loss += ks_gap_loss*ks_gap_weight
            metrics['KS_gap_loss'] += ks_gap_loss.item() * batch.num_graphs
        
        if hasattr(batch, 'E_gap'):
            e_gap_loss = F.mse_loss(outputs['E_gap'], batch.E_gap.view(-1, 1))
            loss += e_gap_loss*e_gap_weight
            metrics['E_gap_loss'] += e_gap_loss.item() * batch.num_graphs

        # If we're using physics constraints with denormalized values
        if physics_loss_weight > 0 and scalers is not None:
            # Denormalize the predictions for physical constraint
            ks_gap_denorm = scalers['KS_gap'].inverse_transform(
                outputs['KS_gap'].detach().cpu().numpy().reshape(-1, 1)
            )
            e_gap_denorm = scalers['E_gap'].inverse_transform(
                outputs['E_gap'].detach().cpu().numpy().reshape(-1, 1)
            )
            
            # Convert back to tensor on the correct device
            ks_gap_denorm = torch.tensor(ks_gap_denorm, device=device).float()
            e_gap_denorm = torch.tensor(e_gap_denorm, device=device).float()


            # Apply physical constraint on denormalized values
            violation = physics_loss_weight * torch.mean(F.relu(ks_gap_denorm - e_gap_denorm)) # Ensure KS_gap <= E_gap *because relu only allows positive values
            loss += violation
        
        # Backward pass and optimization
        if loss > 0:
            loss.backward()
            
            # Monitor gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in all_gradients:
                        all_gradients[name] = []
                    all_gradients[name].append(param.grad.abs().mean().item())
            
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            metrics['count'] += batch.num_graphs
    
    # Calculate average losses
    if metrics['count'] > 0:
        total_loss /= metrics['count']
        metrics['energy_loss'] /= metrics['count']
        metrics['KS_gap_loss'] /= metrics['count']
        metrics['E_gap_loss'] /= metrics['count']
    
    # # Calculate average gradients
    # avg_gradients = {name: sum(values)/len(values) for name, values in all_gradients.items()}
    # max_gradients = {name: max(values) for name, values in all_gradients.items()}
    
    # # Print gradient statistics
    # print("\nGradient Statistics:")
    # for layer_type in ["conv", "energy_predictor", "KS_gap_predictor", "E_gap_predictor"]:
    #     layer_grads = {k: v for k, v in avg_gradients.items() if layer_type in k}
    #     if layer_grads:
    #         print(f"  {layer_type} average gradient magnitude: {sum(layer_grads.values())/len(layer_grads):.5f}")
    #         max_grad = max(max_gradients[k] for k in layer_grads.keys())
    #         print(f"  {layer_type} max gradient magnitude: {max_grad:.5f}")
    
    return total_loss, metrics

# Evaluation function
def evaluate(model, loader, device, scalers=None):
    model.eval()
    predictions = {'energy': [], 'KS_gap': [], 'E_gap': []}
    targets = {'energy': [], 'KS_gap': [], 'E_gap': []}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            # Collect predictions and targets (normalized)
            for key in outputs:
                if hasattr(batch, key):
                    predictions[key].extend(outputs[key].cpu().numpy().flatten())
                    targets[key].extend(batch[key].cpu().numpy().flatten())
    
    # Denormalize predictions and targets for reporting
    denorm_predictions = {}
    denorm_targets = {}
    
    if scalers is not None:
        for key in predictions:
            if key in scalers and len(predictions[key]) > 0:
                pred_array = np.array(predictions[key]).reshape(-1, 1)
                target_array = np.array(targets[key]).reshape(-1, 1)
                denorm_predictions[key] = scalers[key].inverse_transform(pred_array).flatten()
                denorm_targets[key] = scalers[key].inverse_transform(target_array).flatten()
            else:
                denorm_predictions[key] = predictions[key]
                denorm_targets[key] = targets[key]
    else:
        denorm_predictions = predictions
        denorm_targets = targets
    
    # Calculate metrics on denormalized values
    metrics = {}
    for key in denorm_predictions:
        if np.any(denorm_targets[key]):
            pred = np.array(denorm_predictions[key])
            true = np.array(denorm_targets[key])
            mse = np.mean((pred - true) ** 2)
            mae = np.mean(np.abs(pred - true))
            metrics[f'{key}_mse'] = mse
            metrics[f'{key}_mae'] = mae
    
    return metrics, denorm_predictions, denorm_targets

# Main training and evaluation pipeline
# Updated run_training function with normalization


# Updated run_training function with normalization
def run_training(df, test_df=None, hidden_dim=64, batch_size=32, epochs=100, lr=0.001, physics_loss_weight=0.1, save_path="", scalers=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets
    normalized_df, scalers, cm_stats = normalize_data(df)
    train_data = prepare_dataset(normalized_df)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"Prepared dataset with {len(train_data)} samples")
    
    # Normalize test data with same parameters if provided
    test_loader = None
    if test_df is not None:
        # Use the same scalers for test data
        normalized_test_df = test_df.copy()
        for target in ['energy', 'KS_gap', 'E_gap']:
            if target in test_df.columns and target in scalers:
                normalized_test_df.loc[~test_df[target].isna(), target] = scalers[target].transform(
                    test_df.loc[~test_df[target].isna(), target].values.reshape(-1, 1)
                ).flatten()
        
        test_data = prepare_dataset(normalized_test_df)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # Create data loaders
    
    print("Created data loaders")
    
    # Initialize model
    node_dim = 1
    edge_dim = 1
    #hidden_dim = 64
    print("Generating model with hidden dim:", hidden_dim)
    print("Training with epochs:", epochs)
    print("Learning rate:", lr)
    model = GraphNet(node_dim, edge_dim, hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    
    # Store metrics for plotting
    all_train_metrics = {
        'energy_loss': [], 
        'KS_gap_loss': [], 
        'E_gap_loss': [], 
        'total_loss': []
    }
    all_test_metrics = {
        'energy_mse': [], 
        'KS_gap_mse': [], 
        'E_gap_mse': []
    }
    
# Training loop
    for epoch in range(epochs):
        # Train with the scalers
        train_loss, train_metrics = train(model, train_loader, optimizer, device, 
                                         physics_loss_weight, scalers)
        
        # Store training metrics
        all_train_metrics['total_loss'].append(train_loss)
        all_train_metrics['energy_loss'].append(train_metrics['energy_loss'])
        all_train_metrics['KS_gap_loss'].append(train_metrics['KS_gap_loss'])
        all_train_metrics['E_gap_loss'].append(train_metrics['E_gap_loss'])
        
        # Evaluate on test set if available
        if test_loader:
            test_metrics, _, _ = evaluate(model, test_loader, device)
            
            # Store test metrics
            all_test_metrics['energy_mse'].append(test_metrics.get('energy_mse', 0))
            all_test_metrics['KS_gap_mse'].append(test_metrics.get('KS_gap_mse', 0))
            all_test_metrics['E_gap_mse'].append(test_metrics.get('E_gap_mse', 0))
            
            scheduler.step(sum(test_metrics.values()))
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                      f'Energy Loss: {train_metrics["energy_loss"]:.4f}, '
                      f'KS_gap Loss: {train_metrics["KS_gap_loss"]:.4f}, '
                      f'E_gap Loss: {train_metrics["E_gap_loss"]:.4f}, '
                      f'Test MSE: Energy={test_metrics.get("energy_mse", "N/A"):.4f}, '
                      f'KS_gap={test_metrics.get("KS_gap_mse", "N/A"):.4f}, '
                      f'E_gap={test_metrics.get("E_gap_mse", "N/A"):.4f}')
            
            # Save best model
            current_loss = sum(test_metrics.values())
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), 'best_model.pt')
        else:
            scheduler.step(train_loss)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                      f'Energy Loss: {train_metrics["energy_loss"]:.4f}, '
                      f'KS_gap Loss: {train_metrics["KS_gap_loss"]:.4f}, '
                      f'E_gap Loss: {train_metrics["E_gap_loss"]:.4f}')
            
            # Save best model based on training loss
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), 'best_model.pt')

    # Save the final model if save_path is provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model, all_train_metrics, all_test_metrics, scalers


# Function for inference on new data
def predict_properties(model, CM, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert Coulomb matrix to graph
    graph_data = convert_matrix_to_graph(CM)
    
    # Move data to device
    graph_data = graph_data.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(graph_data)
    
    # Extract results
    results = {
        'energy': outputs['energy'].item(),
        'KS_gap': outputs['KS_gap'].item(),
        'E_gap': outputs['E_gap'].item()
    }
    
    return results

# Function to plot training and testing performance
def plot_training_performance(train_metrics, test_metrics=None, save_path=None, title=""):
    plt.figure(figsize=(15, 10))
    
    # Plot training metrics
    epochs = range(1, len(train_metrics['energy_loss']) + 1)
    
    # Subplot 1: Energy losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_metrics['energy_loss'], 'b-', label='Train Energy Loss')
    if test_metrics and 'energy_mse' in test_metrics and test_metrics['energy_mse']:
        plt.plot(epochs, test_metrics['energy_mse'], 'r--', label='Test Energy MSE')
    plt.title('Energy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: KS_gap losses
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_metrics['KS_gap_loss'], 'b-', label='Train KS_gap Loss')
    if test_metrics and 'KS_gap_mse' in test_metrics and test_metrics['KS_gap_mse']:
        plt.plot(epochs, test_metrics['KS_gap_mse'], 'r--', label='Test KS_gap MSE')
    plt.title('KS_gap Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 3: E_gap losses
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_metrics['E_gap_loss'], 'b-', label='Train E_gap Loss')
    if test_metrics and 'E_gap_mse' in test_metrics and test_metrics['E_gap_mse']:
        plt.plot(epochs, test_metrics['E_gap_mse'], 'r--', label='Test E_gap MSE')  
    plt.title('E_gap Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()