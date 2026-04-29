"""
Pile Stiffness Degradation - Training Script
Slot-Attention Transformer Architecture (PDF Section 4.2.1)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SlotAttentionDegradation(nn.Module):
    """
    Slot-Attention Transformer for Pile Stiffness Degradation.
    
    Architecture (PDF Section 4.2.1):
    - 21 Slots: Slot 1 = H0 (initial), Slots 2-21 = Hn (drops)
    - Iterative slot refinement via cross-attn + self-attn
    - LSTM decoder with cross-attention to drop slots
    - Physics constraint: KL/KR only decrease, KLR unconstrained
    """
    
    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21, 
                 max_seq_len=50, dropout=0.1, num_iterations=2):
        super().__init__()
        
        self.num_slots = num_slots
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Learnable slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        
        # Slot refinement layers
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        
        # LSTM decoder
        self.seq_decoder = nn.LSTM(d_model, d_model, 2, batch_first=True, dropout=dropout)
        
        # Decoder cross-attention to drop slots
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_norm = nn.LayerNorm(d_model)
        
        # LSTM hidden state from initial slot
        self.h0_proj = nn.Linear(d_model, d_model * 2)
        self.c0_proj = nn.Linear(d_model, d_model * 2)
        
        # Output heads
        self.initial_proj = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        batch_size = x.size(0)
        
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(batch_size, -1, -1)
        
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        
        slot_initial = slots[:, 0:1, :]
        slot_drops = slots[:, 1:, :]
        
        initial_decoded = self.initial_proj(slot_initial)
        
        init_vec = slot_initial.squeeze(1)
        h0 = self.h0_proj(init_vec).view(batch_size, 2, self.d_model).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(init_vec).view(batch_size, 2, self.d_model).permute(1, 0, 2).contiguous()
        
        drop_agg = slot_drops.mean(dim=1, keepdim=True).expand(-1, seq_len-1, -1)
        decoder_input = drop_agg + self.pos_embed[:, :seq_len-1, :]
        lstm_out, _ = self.seq_decoder(decoder_input, (h0, c0))
        
        refined = self.decoder_cross_attn(lstm_out, slot_drops, slot_drops)[0]
        refined = self.decoder_cross_norm(lstm_out + refined)
        
        raw_drops = self.drop_proj(refined)
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = raw_drops[:, :, 2:3]
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        
        cumulative_drops = torch.cumsum(drops, dim=1)
        stiffness_after_drops = initial_decoded + cumulative_drops
        stiffness_seq = torch.cat([initial_decoded, stiffness_after_drops], dim=1)
        
        return stiffness_seq


def load_and_group_data(excel_path):
    """Load Excel and group by scenario (unique input combinations)."""
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    
    col_map = {'Dp/Lp': 'Dp_Lp', 'KL': 'kl', 'KR': 'kr', 'KLR': 'klr'}
    df.rename(columns=col_map, inplace=True)
    
    input_cols = ['PI', 'Gmax', 'v', 'Dp', 'Tp', 'Lp', 'Ip', 'Dp_Lp']
    output_cols = ['kl', 'kr', 'klr']
    
    # Group by input parameters
    groups = df.groupby(input_cols, sort=False)
    
    X_list, Y_list = [], []
    for name, group in groups:
        outputs = group[output_cols].values
        if len(outputs) < 2:
            continue
        
        # Convert drops to actual values
        initial = outputs[0]
        drops = outputs[1:]
        actual = initial - np.cumsum(drops, axis=0)
        full_seq = np.vstack([initial, actual])
        
        X_list.append(list(name))
        Y_list.append(full_seq)
    
    seq_lengths = [len(y) for y in Y_list]
    print(f"Loaded {len(X_list)} scenarios, seq lengths: {min(seq_lengths)}-{max(seq_lengths)}")
    
    return X_list, Y_list, seq_lengths, input_cols, output_cols


def pad_sequences(Y_list, max_len):
    """Pad sequences to uniform length."""
    padded = []
    for y in Y_list:
        if len(y) < max_len:
            pad = np.tile(y[-1:], (max_len - len(y), 1))
            y = np.vstack([y, pad])
        padded.append(y[:max_len])
    return np.array(padded)


def train_model(X_train, Y_train, X_val, Y_val, max_seq_len, epochs=3000, batch_size=4, lr=0.0005):
    """Train the model with composite loss."""
    model = SlotAttentionDegradation(
        input_size=X_train.shape[1],
        d_model=64, num_heads=4, num_slots=21,
        max_seq_len=max_seq_len, dropout=0.1, num_iterations=2
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, min_lr=1e-6)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 0
    max_patience = 300
    
    print(f"\nTraining: {epochs} epochs, batch={batch_size}, lr={lr}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch, seq_len=max_seq_len)
            
            # Composite loss
            # 1. Sequence loss (Huber for robustness to outliers)
            loss_seq = huber(pred, Y_batch)
            
            # 2. Initial value emphasis (getting the start right matters)
            loss_initial = mse(pred[:, 0, :], Y_batch[:, 0, :]) * 3.0
            
            # 3. Shape loss: match the degradation pattern step-to-step
            diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
            diff_target = Y_batch[:, 1:, :] - Y_batch[:, :-1, :]
            loss_shape = huber(diff_pred, diff_target)
            
            loss = loss_seq + loss_initial + loss_shape
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val), seq_len=max_seq_len)
            val_target = torch.FloatTensor(Y_val)
            val_loss = mse(val_pred, val_target).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr_now:.6f}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model


def main():
    excel_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')
    
    X_list, Y_list, seq_lengths, input_cols, output_cols = load_and_group_data(excel_path)
    max_seq_len = max(seq_lengths)
    
    Y_padded = pad_sequences(Y_list, max_seq_len)
    X_array = np.array(X_list)
    
    # Normalize
    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X_array)
    
    Y_sign = np.sign(Y_padded)
    Y_log = Y_sign * np.log1p(np.abs(Y_padded))
    
    # Per-output scaling
    scaler_Y = RobustScaler()
    Y_flat = Y_log.reshape(-1, 3)
    Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(Y_log.shape)
    
    # Split (index-based to track original values)
    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    Y_train, Y_test = Y_scaled[train_idx], Y_scaled[test_idx]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}, Seq len: {max_seq_len}")
    
    # Train
    model = train_model(X_train, Y_train, X_test, Y_test, max_seq_len)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test), seq_len=max_seq_len).numpy()
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    for i, name in enumerate(output_cols):
        r2 = r2_score(Y_test[:,:,i].flatten(), test_pred[:,:,i].flatten())
        print(f"{name.upper()}: R² = {r2:.4f}")
    print(f"Overall R²: {r2_score(Y_test.flatten(), test_pred.flatten()):.4f}")
    print("="*40)
    
    # Save
    torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, 'pile_model.pth'))
    joblib.dump(scaler_X, os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
    joblib.dump(input_cols, os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
    joblib.dump(max_seq_len, os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
    
    # Save test data for webapp visualization
    test_data = {
        'X_original': X_array[test_idx],
        'Y_original': Y_padded[test_idx],
        'X_scaled': X_test,
        'input_cols': input_cols,
        'output_cols': output_cols,
    }
    joblib.dump(test_data, os.path.join(SCRIPT_DIR, 'test_data.pkl'))
    
    print("\nSaved! Run: python webapp.py")


if __name__ == "__main__":
    main()
