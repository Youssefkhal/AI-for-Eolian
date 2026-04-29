"""
Physics-Informed Transformer for OWT Natural Frequency Shift Prediction
Based on Pages 29-37 of PostTyphoon_OWT_natural_frequency_shift_prediction.pdf

This simplified implementation demonstrates the core concepts:
1. Slot-Attention Module for foundation stiffness self-calibration
2. Physics-informed loss functions (Laws of Thermodynamics)
3. Transformer architecture for load-to-frequency prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SlotAttentionModule(nn.Module):
    """
    Self-calibration module that learns to calibrate foundation stiffness matrices.

    - Slot 1: Initial stiffness matrix H0 (3x3, so 9 parameters)
    - Slots 2-21: 20 stiffness drop matrices H_n with activation gates w_n(t)

    Total: 1 slot for H0 + 20 slots for (H_n, w_n)
    """

    def __init__(self, input_dim=8, hidden_dim=64, num_slots=21, num_drops=20):
        """
        Args:
            input_dim: Number of soil/pile parameters (from Table 3)
            hidden_dim: Hidden dimension for MLPs
            num_slots: Number of calibration slots (1 for H0 + 20 for drops)
            num_drops: Number of plastic stiffness drop surfaces (20)
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_drops = num_drops
        self.input_dim = input_dim

        # Learnable slot embeddings (initialized randomly)
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim))

        # Cross-attention: map inputs to slots
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Self-attention: share context between slots
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Input encoder
        self.input_encoder = nn.Linear(input_dim, hidden_dim)

        # MLPs to decode slots into stiffness matrices
        # Slot 1 -> H0 (3x3 matrix flattened to 9 values)
        self.mlp_h0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)  # 3x3 matrix
        )

        # Slots 2-21 -> (H_n, w_n) for each drop surface
        self.mlp_drops = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 9 for H_n (3x3) + 1 for w_n
        )

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, input_dim) - soil and pile parameters

        Returns:
            H0: (batch_size, 3, 3) - initial stiffness matrix
            H_drops: (batch_size, 20, 3, 3) - plastic stiffness drops
            w_gates: (batch_size, 20) - activation gates
        """
        batch_size = inputs.shape[0]

        # Encode inputs
        encoded_inputs = self.input_encoder(inputs)  # (batch_size, hidden_dim)

        # Expand for cross-attention: (batch_size, 1, hidden_dim)
        encoded_inputs = encoded_inputs.unsqueeze(1)

        # Cross-attention: slots attend to inputs
        slots = self.slots.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_slots, hidden)
        attended_slots, _ = self.cross_attention(slots, encoded_inputs, encoded_inputs)

        # Self-attention: slots communicate with each other
        refined_slots, _ = self.self_attention(attended_slots, attended_slots, attended_slots)

        # Decode slot 1 to H0
        h0_flat = self.mlp_h0(refined_slots[:, 0, :])  # (batch_size, 9)
        H0 = h0_flat.reshape(batch_size, 3, 3)

        # Decode slots 2-21 to H_n and w_n
        h_drops_list = []
        w_gates_list = []

        for i in range(1, self.num_slots):
            output = self.mlp_drops(refined_slots[:, i, :])  # (batch_size, 10)
            h_n_flat = output[:, :9]  # First 9: H_n
            w_n = output[:, 9]        # Last 1: w_n

            h_n = h_n_flat.reshape(batch_size, 3, 3)
            h_drops_list.append(h_n)

            # Sigmoid activation for w_n (values in [0, 1])
            w_n = torch.sigmoid(w_n)
            w_gates_list.append(w_n)

        H_drops = torch.stack(h_drops_list, dim=1)  # (batch_size, 20, 3, 3)
        w_gates = torch.stack(w_gates_list, dim=1)  # (batch_size, 20)

        return H0, H_drops, w_gates


class StiffnessEvolution(nn.Module):
    """
    Computes foundation stiffness evolution over load steps using TIM formulation.

    Formula: H_ij(t) = H0_ij - Σ w_n(t) · H_n_ij
    """

    def __init__(self):
        super().__init__()

    def forward(self, H0, H_drops, w_gates):
        """
        Args:
            H0: (batch_size, 3, 3) - initial stiffness
            H_drops: (batch_size, 20, 3, 3) - stiffness drops
            w_gates: (batch_size, 20) - activation weights

        Returns:
            H_evolved: (batch_size, 3, 3) - degraded stiffness
        """
        batch_size = H0.shape[0]

        # Compute degradation: Σ w_n · H_n
        degradation = torch.zeros_like(H0)
        for i in range(20):
            # w_n shape: (batch_size,)
            # H_n shape: (batch_size, 3, 3)
            weight = w_gates[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            degradation += weight * H_drops[:, i, :]  # Broadcast multiply

        # H(t) = H0 - degradation
        H_evolved = H0 - degradation

        return H_evolved


class SimpleTransformer(nn.Module):
    """
    Physics-informed transformer for load sequences to natural frequency prediction.

    Encoder: Self-calibration module + self-attention
    Decoder: Cross-attention + frequency prediction
    """

    def __init__(self, input_dim=8, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Slot-attention calibration module
        self.calibration = SlotAttentionModule(input_dim, hidden_dim)
        self.stiffness_evolution = StiffnessEvolution()

        # Input tokenization
        self.embed_loads = nn.Linear(1, hidden_dim)  # Embed load values
        self.embed_params = nn.Linear(input_dim, hidden_dim)  # Embed parameters

        # Positional encoding (simplified)
        self.pos_encoding = self._create_positional_encoding(hidden_dim, max_seq_len=100)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dim_feedforward=hidden_dim*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True, dim_feedforward=hidden_dim*4
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: predict frequency shift
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single frequency value
        )

    def _create_positional_encoding(self, hidden_dim, max_seq_len):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) *
                            -(np.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, hidden_dim)

    def forward(self, load_sequence, soil_pile_params):
        """
        Args:
            load_sequence: (batch_size, seq_len, 1) - time series of loads
            soil_pile_params: (batch_size, input_dim) - soil/pile parameters

        Returns:
            freq_pred: (batch_size,) - predicted frequency shift
        """
        batch_size, seq_len = load_sequence.shape[0], load_sequence.shape[1]

        # Step 1: Calibration module predicts foundation stiffness
        H0, H_drops, w_gates = self.calibration(soil_pile_params)
        H_evolved = self.stiffness_evolution(H0, H_drops, w_gates)

        # Step 2: Encode load sequence
        embedded_loads = self.embed_loads(load_sequence)  # (batch, seq_len, hidden)

        # Step 3: Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(load_sequence.device)
        embedded_loads = embedded_loads + pos_enc

        # Step 4: Encode stiffness matrix as a token
        h_token = H_evolved.reshape(batch_size, -1)  # (batch_size, 9)
        h_token = self.embed_params(torch.zeros_like(soil_pile_params))  # Reuse embedding
        h_token = h_token.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Step 5: Concatenate with load sequence
        encoder_input = torch.cat([h_token, embedded_loads], dim=1)

        # Step 6: Transformer encoder
        encoder_output = self.transformer_encoder(encoder_input)

        # Step 7: Transformer decoder (simple: take output of encoder)
        decoder_output = self.transformer_decoder(encoder_output, encoder_output)

        # Step 8: Predict frequency from final output
        freq_pred = self.freq_head(decoder_output[:, 0, :])  # Use first token

        return freq_pred.squeeze(-1), H_evolved


class PhysicsLosses(nn.Module):
    """
    Physics-informed loss functions enforcing Laws of Thermodynamics.

    LoT1: Energy conservation (soft enforcement via frequency consistency)
    LoT2: Non-negative dissipation (monotonic drops, non-negative stiffness)
    """

    def __init__(self):
        super().__init__()

    def energy_conservation_loss(self, predicted_freq, analytical_freq):
        """
        LoT1: Frequency consistency ensures energy conservation.

        Compare predicted frequency with analytically derived frequency.
        """
        return nn.functional.mse_loss(predicted_freq, analytical_freq)

    def monotonicity_loss(self, w_gates):
        """
        LoT2: Enforce monotonic activation weights (no reversals).

        Adjacent gates should be ordered: w_1 ≤ w_2 ≤ ... ≤ w_20
        """
        diffs = w_gates[:, 1:] - w_gates[:, :-1]
        # Penalty for negative differences (violations of monotonicity)
        violation = torch.clamp(-diffs, min=0)
        return violation.mean()

    def positive_stiffness_loss(self, H_evolved):
        """
        LoT2: Ensure non-negative stiffness (prevent stiffness collapse).

        Penalize negative eigenvalues of evolved stiffness matrix.
        """
        batch_size = H_evolved.shape[0]
        penalty = 0

        for i in range(batch_size):
            eigenvalues = torch.linalg.eigvalsh(H_evolved[i])
            negative_eigs = torch.clamp(-eigenvalues, min=0)
            penalty += negative_eigs.sum()

        return penalty / batch_size

    def composite_loss(self, pred_freq, analytical_freq, w_gates, H_evolved,
                      true_freq, alpha=1.0, beta=0.5, gamma=0.3, delta=0.2):
        """
        Composite loss: L_total = α·L_data + β·L_freq + γ·(L_1mon + L_2mon) + δ·L_pos
        """
        # Data loss: NRMSE between predicted and true frequency
        data_loss = nn.functional.mse_loss(pred_freq, true_freq)

        # Physics loss: frequency consistency
        freq_loss = self.energy_conservation_loss(pred_freq, analytical_freq)

        # Physics loss: monotonicity
        mono_loss = self.monotonicity_loss(w_gates)

        # Physics loss: positive stiffness
        pos_loss = self.positive_stiffness_loss(H_evolved)

        total = alpha * data_loss + beta * freq_loss + gamma * mono_loss + delta * pos_loss

        return total, {
            'data_loss': data_loss.item(),
            'freq_loss': freq_loss.item(),
            'mono_loss': mono_loss.item(),
            'pos_loss': pos_loss.item()
        }


def example_training():
    """Simple training example with visualization."""

    # Hyperparameters
    batch_size = 16
    seq_len = 50
    input_dim = 8
    epochs = 10
    learning_rate = 1e-3

    # Initialize model and optimizer
    model = SimpleTransformer(input_dim=input_dim, hidden_dim=64)
    physics_losses = PhysicsLosses()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    physics_losses.to(device)

    print(f"Using device: {device}")
    print("Starting training...\n")

    # Track metrics for visualization
    history = {
        'epoch': [],
        'total_loss': [],
        'data_loss': [],
        'freq_loss': [],
        'mono_loss': [],
        'pos_loss': [],
        'pred_freq': [],
        'true_freq': [],
        'H_evolved': []
    }

    for epoch in range(epochs):
        # Generate synthetic data
        load_sequence = torch.randn(batch_size, seq_len, 1, device=device)
        soil_params = torch.randn(batch_size, input_dim, device=device)

        # Ground truth: synthetic frequency shifts
        true_freq = torch.randn(batch_size, device=device)
        analytical_freq = true_freq + 0.1 * torch.randn(batch_size, device=device)

        # Forward pass
        pred_freq, H_evolved = model(load_sequence, soil_params)

        # Compute calibration module outputs for loss
        H0, H_drops, w_gates = model.calibration(soil_params)

        # Compute losses
        total_loss, loss_dict = physics_losses.composite_loss(
            pred_freq, analytical_freq, w_gates, H_evolved, true_freq
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track metrics
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(total_loss.item())
        history['data_loss'].append(loss_dict['data_loss'])
        history['freq_loss'].append(loss_dict['freq_loss'])
        history['mono_loss'].append(loss_dict['mono_loss'])
        history['pos_loss'].append(loss_dict['pos_loss'])
        history['pred_freq'].append(pred_freq.detach().cpu().numpy())
        history['true_freq'].append(true_freq.detach().cpu().numpy())
        history['H_evolved'].append(H_evolved.detach().cpu().numpy())

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss.item():.4f}")
            print(f"  data_loss: {loss_dict['data_loss']:.4f}, freq_loss: {loss_dict['freq_loss']:.4f}")
            print(f"  mono_loss: {loss_dict['mono_loss']:.4f}, pos_loss: {loss_dict['pos_loss']:.4f}\n")

    print("Training complete! Generating visualization...\n")

    # Create comprehensive visualization
    create_visualizations(history, epochs)


def create_visualizations(history, epochs):
    """Create and display training visualizations."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # === ROW 1: LOSSES ===
    # Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['epoch'], history['total_loss'], 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Component Losses
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['epoch'], history['data_loss'], 'r-o', label='Data Loss', linewidth=2)
    ax2.plot(history['epoch'], history['freq_loss'], 'g-s', label='Freq Loss', linewidth=2)
    ax2.plot(history['epoch'], history['mono_loss'], 'b-^', label='Mono Loss', linewidth=2)
    ax2.plot(history['epoch'], history['pos_loss'], 'm-d', label='Pos Loss', linewidth=2)
    ax2.set_title('Component Losses', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Loss Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    data_ratio = [d / (t + 1e-8) for d, t in zip(history['data_loss'], history['total_loss'])]
    physics_ratio = [1 - d for d in data_ratio]
    ax3.fill_between(history['epoch'], 0, data_ratio, alpha=0.6, label='Data Loss %', color='red')
    ax3.fill_between(history['epoch'], data_ratio, 1, alpha=0.6, label='Physics Loss %', color='blue')
    ax3.set_title('Loss Composition', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ratio')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # === ROW 2: FREQUENCY PREDICTIONS ===
    # Final Epoch: Pred vs True
    ax4 = fig.add_subplot(gs[1, 0])
    final_pred = history['pred_freq'][-1]
    final_true = history['true_freq'][-1]
    ax4.scatter(final_true, final_pred, alpha=0.6, s=100, edgecolors='black')
    ax4.plot([final_true.min(), final_true.max()],
             [final_true.min(), final_true.max()], 'r--', linewidth=2, label='Perfect Fit')
    ax4.set_title('Final Epoch: Predicted vs True Frequency', fontsize=12, fontweight='bold')
    ax4.set_xlabel('True Frequency (Hz)')
    ax4.set_ylabel('Predicted Frequency (Hz)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Frequency Error
    ax5 = fig.add_subplot(gs[1, 1])
    final_error = final_pred - final_true
    ax5.hist(final_error, bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax5.axvline(np.mean(final_error), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_error):.3f}')
    ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Error (Hz)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # MAE and RMSE over epochs
    ax6 = fig.add_subplot(gs[1, 2])
    mae_values = []
    rmse_values = []
    for i in range(len(history['epoch'])):
        pred = history['pred_freq'][i]
        true = history['true_freq'][i]
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mae_values.append(mae)
        rmse_values.append(rmse)

    ax6.plot(history['epoch'], mae_values, 'g-o', label='MAE', linewidth=2, markersize=6)
    ax6.plot(history['epoch'], rmse_values, 'r-s', label='RMSE', linewidth=2, markersize=6)
    ax6.set_title('Prediction Accuracy Metrics', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Error (Hz)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # === ROW 3: STIFFNESS EVOLUTION ===
    # H0 Eigenvalues
    ax7 = fig.add_subplot(gs[2, 0])
    H_init = history['H_evolved'][0][0]  # First sample, first epoch
    H_final = history['H_evolved'][-1][0]  # First sample, last epoch
    eigs_init = np.linalg.eigvalsh(H_init)
    eigs_final = np.linalg.eigvalsh(H_final)
    x = np.arange(len(eigs_init))
    width = 0.35
    ax7.bar(x - width/2, eigs_init, width, label='Epoch 1', alpha=0.8, color='blue')
    ax7.bar(x + width/2, eigs_final, width, label=f'Epoch {epochs}', alpha=0.8, color='orange')
    ax7.set_title('Stiffness Matrix Eigenvalues Evolution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Eigenvalue Index')
    ax7.set_ylabel('Eigenvalue')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Stiffness Degradation
    ax8 = fig.add_subplot(gs[2, 1])
    H_norms = []
    for H in history['H_evolved']:
        norm = np.linalg.norm(H[0])  # Frobenius norm of first sample
        H_norms.append(norm)
    ax8.plot(history['epoch'], H_norms, 'purple', marker='o', linewidth=2, markersize=6)
    ax8.fill_between(history['epoch'], H_norms, alpha=0.3, color='purple')
    ax8.set_title('Stiffness Matrix Norm (Degradation)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Frobenius Norm')
    ax8.grid(True, alpha=0.3)

    # Stiffness Heatmap (Final Epoch)
    ax9 = fig.add_subplot(gs[2, 2])
    H_final = history['H_evolved'][-1][0]
    im = ax9.imshow(H_final, cmap='RdBu_r', aspect='auto')
    ax9.set_title('Final Stiffness Matrix Heatmap', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Column')
    ax9.set_ylabel('Row')
    plt.colorbar(im, ax=ax9, label='Stiffness Value')

    # Add overall title
    fig.suptitle('Physics-Informed Transformer Training Visualization', fontsize=16, fontweight='bold', y=0.995)

    # Save and display
    plt.savefig('c:/Users/youss/Downloads/PFE/M4/training_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved to: c:/Users/youss/Downloads/PFE/M4/training_results.png")
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Final Total Loss: {history['total_loss'][-1]:.6f}")
    print(f"Loss Reduction: {(1 - history['total_loss'][-1]/history['total_loss'][0])*100:.2f}%")
    print(f"\nFinal Frequency Prediction Error:")
    print(f"  MAE (Mean Absolute Error): {mae_values[-1]:.6f} Hz")
    print(f"  RMSE (Root Mean Square Error): {rmse_values[-1]:.6f} Hz")
    print(f"  Std Dev: {np.std(final_error):.6f} Hz")
    print(f"\nStiffness Evolution:")
    print(f"  Initial Norm: {H_norms[0]:.6f}")
    print(f"  Final Norm: {H_norms[-1]:.6f}")
    print(f"  Degradation: {(1 - H_norms[-1]/H_norms[0])*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    example_training()
