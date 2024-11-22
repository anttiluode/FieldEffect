import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FieldEnhancedLayer(nn.Module):
    def __init__(self, in_features, out_features, field_size=(32, 32)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.field_size = field_size
        
        # Neural weights
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Field parameters
        self.field_coupling = nn.Parameter(torch.randn(1, 1, *field_size) * 0.01)
        self.phase_sensitivity = nn.Parameter(torch.ones(out_features) * 0.5)
        
        # Make field parameters learnable instead of using buffers
        self.field_magnitude = nn.Parameter(torch.zeros(1, 1, *field_size))
        self.field_phase = nn.Parameter(torch.zeros(1, 1, *field_size))
        
        # Projection matrix
        self.projection = nn.Parameter(
            torch.randn(self.in_features, field_size[0] * field_size[1]) * 0.01
        )
        
    def compute_field_effect(self, x):
        batch_size = x.size(0)
        
        # Project input to field space
        field_input = torch.mm(x, self.projection)
        field_input = field_input.view(batch_size, 1, *self.field_size)
        
        # Compute field perturbation
        field_perturbation = field_input.mean(dim=0, keepdim=True) * self.field_coupling
        
        # Update magnitude (without in-place operations)
        new_magnitude = torch.sigmoid(self.field_magnitude + 0.1 * field_perturbation)
        
        # Update phase (without in-place operations)
        phase_delta = 0.1 * torch.tanh(field_perturbation)
        new_phase = torch.remainder(self.field_phase + phase_delta, 2 * np.pi)
        
        # Compute field effect
        field_effect = new_magnitude * torch.sin(new_phase)
        
        return field_effect, new_magnitude, new_phase
        
    def forward(self, x):
        # Standard neural computation
        standard_output = torch.mm(x, self.weights.t()) + self.bias
        
        # Compute field effect
        field_effect, new_magnitude, new_phase = self.compute_field_effect(x)
        
        # Update field states
        self.field_magnitude.data = new_magnitude.data
        self.field_phase.data = new_phase.data
        
        # Pool field effect to match output size
        field_influence = nn.functional.adaptive_avg_pool2d(
            field_effect, (self.out_features, 1)
        ).squeeze(-1).squeeze(0)
        
        # Modulate output with field effect
        output = standard_output * (1 + self.phase_sensitivity * field_influence)
        
        return output, field_effect

class FieldEnhancedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, field_size=(32, 32)):
        super().__init__()
        self.layer1 = FieldEnhancedLayer(input_size, hidden_size, field_size)
        self.layer2 = FieldEnhancedLayer(hidden_size, output_size, field_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # First layer
        x, field1 = self.layer1(x)
        x = self.activation(x)
        
        # Second layer
        x, field2 = self.layer2(x)
        
        # Collect field info
        field_info = {
            'layer1_field': field1.detach(),
            'layer2_field': field2.detach(),
            'layer1_magnitude': self.layer1.field_magnitude.detach(),
            'layer2_magnitude': self.layer2.field_magnitude.detach(),
            'layer1_phase': self.layer1.field_phase.detach(),
            'layer2_phase': self.layer2.field_phase.detach()
        }
        
        return x, field_info

class StandardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.layer2(x)

def train_and_compare(eeg_data, hidden_vectors, field_size=(32, 32), n_epochs=100):
    print("Preparing data and models...")
    
    # Convert to tensors
    X = torch.tensor(hidden_vectors, dtype=torch.float32)
    y = torch.tensor(eeg_data, dtype=torch.float32)
    
    # Initialize networks
    input_size = hidden_vectors.shape[1]
    hidden_size = min(128, input_size * 2)
    output_size = eeg_data.shape[1]
    
    print(f"Network architecture: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Field size: {field_size}")
    
    field_net = FieldEnhancedNetwork(input_size, hidden_size, output_size, field_size)
    standard_net = StandardNetwork(input_size, hidden_size, output_size)
    
    # Training setup
    criterion = nn.MSELoss()
    field_opt = torch.optim.Adam(field_net.parameters(), lr=0.001)
    standard_opt = torch.optim.Adam(standard_net.parameters(), lr=0.001)
    
    history = {
        'field_losses': [],
        'standard_losses': [],
        'field_states': []
    }
    
    print("Starting training...")
    try:
        for epoch in range(n_epochs):
            # Train field network
            field_opt.zero_grad()
            field_output, field_info = field_net(X)
            field_loss = criterion(field_output, y)
            field_loss.backward()
            field_opt.step()
            
            history['field_losses'].append(field_loss.item())
            
            # Train standard network
            standard_opt.zero_grad()
            standard_output = standard_net(X)
            standard_loss = criterion(standard_output, y)
            standard_loss.backward()
            standard_opt.step()
            
            history['standard_losses'].append(standard_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Field Loss = {field_loss.item():.6f}, Standard Loss = {standard_loss.item():.6f}")
                history['field_states'].append({
                    'epoch': epoch,
                    'states': {k: v.cpu().numpy() for k, v in field_info.items()}
                })
                
        improvement = ((standard_loss.item() - field_loss.item()) / standard_loss.item())
        print(f"\nTraining completed!")
        print(f"Final field network loss: {field_loss.item():.6f}")
        print(f"Final standard network loss: {standard_loss.item():.6f}")
        print(f"Improvement: {improvement*100:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
        
    return history, field_net, standard_net

def visualize_results(history):
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Learning curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history['field_losses'], label='Field-Enhanced')
    ax1.plot(history['standard_losses'], label='Standard')
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Final field states
    final_states = history['field_states'][-1]['states']
    
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(final_states['layer1_magnitude'][0,0], cmap='viridis')
    ax2.set_title('Layer 1 Field Magnitude')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(final_states['layer1_phase'][0,0], cmap='hsv')
    ax3.set_title('Layer 1 Field Phase')
    plt.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    im4 = ax4.imshow(final_states['layer1_field'][0,0], cmap='RdBu')
    ax4.set_title('Layer 1 Field Effect')
    plt.colorbar(im4, ax=ax4)
    
    # Improvement over time
    ax5 = fig.add_subplot(gs[2, 1])
    improvements = [(s - f) / s for f, s in 
                   zip(history['standard_losses'], history['field_losses'])]
    ax5.plot(improvements)
    ax5.set_title('Relative Improvement')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Improvement Factor')
    
    plt.tight_layout()
    return fig

def run_test(n_samples=1000, input_size=64, output_size=32):
    print("Generating synthetic data...")
    
    # Create synthetic data
    hidden_vectors = np.random.randn(n_samples, input_size)
    eeg_data = np.random.randn(n_samples, output_size)
    
    # Fixed field size
    field_size = (32, 32)
    
    print("Running comparison test...")
    history, field_net, standard_net = train_and_compare(eeg_data, hidden_vectors, field_size)
    
    print("Visualizing results...")
    fig = visualize_results(history)
    plt.show()
    
    return history, field_net, standard_net

if __name__ == "__main__":
    print("Starting field-enhanced neural computation test...")
    history, field_net, standard_net = run_test()
    print("Test completed!")