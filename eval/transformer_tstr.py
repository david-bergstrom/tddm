import numpy as np
import torch
from torch import nn


class SimpleTransformerTSTR(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, pred_window=10, max_seq_len=1000):
        super().__init__()
        self.pred_window = pred_window
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, feature_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, feature_dim] - input sequence
        Returns:
            predictions: [batch_size, pred_window, feature_dim] - predicted next steps
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x_proj = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        x_proj = x_proj + pos_enc
        
        # Create target sequence for autoregressive prediction
        # Start with the last input step, then predict pred_window steps
        target_seq = torch.zeros(batch_size, self.pred_window, self.d_model, device=x.device)
        
        # Initialize first target with last input step
        target_seq[:, 0] = x_proj[:, -1]  # Use last input as first target
        
        # Create causal mask for target sequence
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_window).to(x.device)
        
        # Autoregressive prediction
        predictions = []
        
        for step in range(self.pred_window):
            # Current target sequence up to this step
            current_target = target_seq[:, :step+1]  # [batch, step+1, d_model]
            
            # Add positional encoding to target
            if step == 0:
                target_pos_enc = self.pos_encoding[seq_len:seq_len+1].unsqueeze(0)
            else:
                target_pos_enc = self.pos_encoding[seq_len:seq_len+step+1].unsqueeze(0)
            current_target = current_target + target_pos_enc
            
            # Apply transformer decoder
            current_mask = tgt_mask[:step+1, :step+1] if step > 0 else None
            
            output = self.transformer(
                tgt=current_target,
                memory=x_proj,
                tgt_mask=current_mask
            )  # [batch, step+1, d_model]
            
            # Get prediction for current step
            current_pred = self.output_proj(output[:, -1])  # [batch, feature_dim]
            predictions.append(current_pred)
            
            # Update target sequence for next step (teacher forcing during training)
            if step < self.pred_window - 1:
                target_seq[:, step+1] = self.input_proj(current_pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch, pred_window, feature_dim]
        
        return predictions


def transformer_tstr_scaled_diff(original, generated, pred_window=10, seed=0, minibatches=3000, batch_size=64):
    """
    Train on synthetic, test on real: train a small transformer on the generated windows to predict the
    last `pred_window` of 128 points, and report its error on every original window (MAE in
    city-normalised units). 3000 minibatches of 64, Adam (lr 1e-3, weight decay 1e-4), StepLR. In detail:
      - inputs are the observed positions minus the last observed position, divided by the standard
        deviation of the generated per-step differences (`scale`), so the predictor sees shapes, not
        absolute positions in the city;
      - outputs are scaled per-step displacements, summed onto the last observed position;
      - training is seeded (torch and the minibatch sampler).
    The predictor trains on the whole generated set (sampled with replacement).

    Args:
        original: Real windows, array or tensor of shape (N, seq_len, 2), city-normalised
        generated: Generated windows, array or tensor of shape (M, seq_len, 2), same frame

    Returns:
        Tensor of N per-window MAEs (mean over the predicted steps and both axes)
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generated = np.asarray(generated, dtype=np.float32)
    scale = float(np.diff(generated, axis=1).std())  # numpy (population) std
    train = torch.from_numpy(generated).cuda()
    model = SimpleTransformerTSTR(train.shape[-1], d_model=32, nhead=4, num_layers=2, pred_window=pred_window).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    def predict(x):
        last = x[:, -1:]
        return last + torch.cumsum(model((x - last) / scale), dim=1) * scale

    model.train()
    for _ in range(minibatches):
        batch = train[torch.from_numpy(rng.integers(0, len(train), batch_size)).cuda()]
        x, y = batch[:, :-pred_window], batch[:, -pred_window:]
        optimizer.zero_grad()
        loss = torch.abs(predict(x) - y).sum() / scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    evaluation = torch.as_tensor(np.asarray(original, dtype=np.float32))
    losses = []
    with torch.no_grad():
        for i in range(0, len(evaluation), 4096):
            b = evaluation[i:i + 4096].cuda()
            losses.append(torch.abs(predict(b[:, :-pred_window]) - b[:, -pred_window:]).mean(dim=(1, 2)).cpu())
    return torch.cat(losses)
