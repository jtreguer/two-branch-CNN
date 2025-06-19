import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoBranchCNN(nn.Module):
    def __init__(self, hsi_bands=224, msi_bands=4, patch_size=31):
        """
        Two-branch CNN for HSI-MSI fusion as described in the paper.
        
        Args:
            hsi_bands (int): Number of bands in the HSI input
            msi_bands (int): Number of bands in the MSI input
            patch_size (int): Size of the MSI patch (default 31x31 as in paper)
        """
        super(TwoBranchCNN, self).__init__()
        
        # HSI branch parameters (1D convolutions)
        self.hsi_branch = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=45, stride=1, padding=22),  # kernel_size=45x1 # CHECK PADDING !!!
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=45, stride=1, padding=22),
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=45, stride=1, padding=22),
            nn.ReLU()
        )
        
        # MSI branch parameters (2D convolutions)
        self.msi_branch = nn.Sequential(
            nn.Conv2d(msi_bands, 30, kernel_size=10, stride=1, padding=5),  # kernel_size=10x10 # CHECK PADDING !!!
            nn.ReLU(),
            nn.Conv2d(30, 30, kernel_size=10, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(30, 30, kernel_size=10, stride=1, padding=5),
            nn.ReLU()
        )
        
        # Calculate the size of features after convolutions
        # For HSI branch: output is (batch_size, 20, hsi_bands)
        # For MSI branch: output is (batch_size, 30, patch_size, patch_size)
        
        # Flatten the features
        self.hsi_flatten = nn.Flatten()
        self.msi_flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(20 * hsi_bands + 30 * patch_size * patch_size, 450),
            nn.ReLU(),
            nn.Linear(450, 450),
            nn.ReLU(),
            nn.Linear(450, hsi_bands))  # Output is the reconstructed HR HSI spectrum
        
    def forward(self, hsi_input, msi_input):
        """
        Forward pass of the network.
        
        Args:
            hsi_input (torch.Tensor): LR HSI spectrum of shape (batch_size, 1, hsi_bands)
            msi_input (torch.Tensor): HR MSI patch of shape (batch_size, msi_bands, patch_size, patch_size)
            
        Returns:
            torch.Tensor: Reconstructed HR HSI spectrum of shape (batch_size, hsi_bands)
        """
        # Process HSI input (1D signal)
        # Add channel dimension if not present (batch_size, hsi_bands) -> (batch_size, 1, hsi_bands)
        if hsi_input.dim() == 2:
            hsi_input = hsi_input.unsqueeze(1)
            
        hsi_features = self.hsi_branch(hsi_input)
        hsi_features = self.hsi_flatten(hsi_features)
        
        # Process MSI input
        msi_features = self.msi_branch(msi_input)
        msi_features = self.msi_flatten(msi_features)
        
        # Concatenate features
        combined_features = torch.cat((hsi_features, msi_features), dim=1)
        
        # Fully connected layers
        output = self.fc_layers(combined_features)
        
        return output

# Example usage
if __name__ == "__main__":
    # Hyperparameters from the paper
    hsi_bands = 224 #162  # Number of bands in HSI (AVIRIS data after removing noisy bands)
    msi_bands = 4    # Number of bands in MSI (Landsat-7)
    patch_size = 31   # Size of MSI patch
    
    # Create model
    model = TwoBranchCNN(hsi_bands=hsi_bands, msi_bands=msi_bands, patch_size=patch_size)
    
    # Example inputs
    batch_size = 4
    hsi_input = torch.randn(batch_size, 1, hsi_bands)  # LR HSI spectrum
    msi_input = torch.randn(batch_size, msi_bands, patch_size, patch_size)  # HR MSI patch
    
    # Forward pass
    output = model(hsi_input, msi_input)
    print(f"Input HSI shape: {hsi_input.shape}")
    print(f"Input MSI shape: {msi_input.shape}")
    print(f"Output shape: {output.shape}")