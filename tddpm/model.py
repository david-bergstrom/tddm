import torch
from torch import nn
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    


class TransformerDenoiser(nn.Module):
    def __init__(self, dim_data, dim_embedding, seq_len, n_heads = 4, n_layers = 4, final_predictor=None, n_decoder_layers=1, heatmap_resolution=64):
        super().__init__()

        if final_predictor is not None:
            raise NotImplementedError("final_predictor is deprecated. Exiting.")

        self.channels = dim_data  # Part of being compatible with DDPM framework
        self.self_condition = False  # Part of being compatible with DDPM framework

        self.noisy_trajectory_token = nn.parameter.Parameter(torch.rand(dim_embedding))
        self.heatmap_token = nn.parameter.Parameter(torch.rand(dim_embedding))
        self.denoising_step_token = nn.parameter.Parameter(torch.rand(dim_embedding))

        self.pos_emb = SinusoidalPositionEmbeddings(dim_embedding)

        self.time_step_embedding = self.pos_emb(torch.arange(seq_len))

        self.denoising_step_emb = SinusoidalPositionEmbeddings(dim_embedding*(1+self.channels))

        # Compute heatmap patch size and grid based on resolution
        # Use smaller patches for lower resolutions to maintain token count
        # 16x16 patches for 128x128, 8x8 for 64x64, 4x4 for 32x32 and 16x16
        self.heatmap_resolution = heatmap_resolution
        self.patch_size = max(4, heatmap_resolution // 8)
        self.heatmap_grid_size = heatmap_resolution // self.patch_size

        indices = np.indices([self.heatmap_grid_size, self.heatmap_grid_size]).reshape(2, -1)
        scaled_x_indices = 100 * indices[0] / self.heatmap_grid_size
        scaled_y_indices = 100 * indices[1] / self.heatmap_grid_size
        self.heatmap_x_embedding = self.pos_emb(torch.tensor(scaled_x_indices))
        self.heatmap_y_embedding = self.pos_emb(torch.tensor(scaled_y_indices))

        pixels_per_patch = self.patch_size ** 2
        self.heatmap_intensity_embedding = nn.Linear(pixels_per_patch, dim_embedding)

        self.dim_embedding = dim_embedding
        self.seq_len = seq_len

        n_heads_enc = n_heads
        n_layers_enc = n_layers
        actual_dim_emb = (self.channels+2)*dim_embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=actual_dim_emb, nhead=n_heads_enc, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_enc)

        if n_decoder_layers == 1:
            self.trajectory_decoder = nn.Linear(actual_dim_emb, dim_data)
        else:
            layers = []

            current_emb_size = actual_dim_emb

            for i in range(n_decoder_layers - 1):
                layers.append(nn.Linear(current_emb_size, current_emb_size // 2))
                layers.append(nn.ReLU())
                current_emb_size = current_emb_size // 2

            layers.append(nn.Linear(current_emb_size, dim_data))
            self.trajectory_decoder = nn.Sequential(*layers)
        
        self.heatmap_decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),  # 48x48
            nn.Conv2d(dim_embedding*4, dim_embedding*2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),  # 96x96
            nn.Conv2d(dim_embedding*2, dim_embedding, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),  # 192x192
            nn.Conv2d(dim_embedding, 1, kernel_size=1, padding='same'),
            nn.ReLU()
        )

    def forward(self, noisy_traj, time, x_self_cond, conditional = None, predict_conditional=False):
        batch_size = noisy_traj.shape[0]
        
        # ic(noisy_traj.shape, time.shape, conditional.shape)

        # ic| x.shape: torch.Size([8, 2, 128])
        #     time.shape: torch.Size([8])
        #     conditional.shape: torch.Size([8, 1, 64, 64])

        embs = []

        for i in range(self.channels):
            i_emb = self.pos_emb(noisy_traj[:, i, :].reshape(-1) * 100)
            i_emb = i_emb.reshape(batch_size, self.seq_len, self.dim_embedding)
            # ic(i_emb.shape)
            embs.append(i_emb)

        # ic(x_emb.shape, y_emb.shape, self.time_step_embedding.shape)

        # ic| x_emb.shape: torch.Size([8, 128, 32])
        #     y_emb.shape: torch.Size([8, 128, 32])
        #     self.time_step_embedding.shape: torch.Size([128, 32])

        time_step_embedding = self.time_step_embedding.unsqueeze(0).expand(embs[0].shape).to(noisy_traj.device)
        noisy_trajectory_token = self.noisy_trajectory_token.reshape(1, 1, self.dim_embedding).expand(embs[0].shape)

        # ic(x_emb.shape, y_emb.shape, time_step_embedding.shape, noisy_trajectory_token.shape,
        #    x_emb.device, y_emb.device, time_step_embedding.device, noisy_trajectory_token.device)

        # ic| x_emb.shape: torch.Size([8, 128, 32])
        #     y_emb.shape: torch.Size([8, 128, 32])
        #     time_step_embedding.shape: torch.Size([8, 128, 32])
        #     noisy_trajectory_token.shape: torch.Size([8, 128, 32])
        #     x_emb.device: device(type='cuda', index=0)
        #     y_emb.device: device(type='cuda', index=0)
        #     time_step_embedding.device: device(type='cuda', index=0)
        #     noisy_trajectory_token.device: device(type='cuda', index=0)

        noisy_trajectory_input = torch.cat(embs + [time_step_embedding, noisy_trajectory_token], axis=2)

        # ic(noisy_trajectory_input.shape)  # ic| noisy_trajectory_input.shape: torch.Size([8, 128, 128])

        # ic(conditional.shape)

        # _, _, W, H = conditional.shape
        # B = W*H  # number of cells in heatmap
        

        if conditional is not None:
            kernel_size = self.patch_size
            stride = self.patch_size

            # ic(conditional.shape)
            patches = conditional.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
            # ic(patches.shape)  # batch_size x 64 x 8 x 8 (64 patches of the image, each of the size of 8x8)
            patches = patches.reshape(patches.size(0), patches.size(1), -1)
            # ic(patches.shape)

            intensity_emb = self.heatmap_intensity_embedding(patches)
            # ic(intensity_emb.shape)

            # Number of blocks
            B = patches.size(1)

            # Expand embeddings to match batch size and number of patches
            batch_size_hm = intensity_emb.shape[0]
            heatmap_x_embedding = self.heatmap_x_embedding.unsqueeze(0).expand(batch_size_hm, -1, -1).to(conditional.device)
            heatmap_y_embedding = self.heatmap_y_embedding.unsqueeze(0).expand(batch_size_hm, -1, -1).to(conditional.device)

            heatmap_token = self.heatmap_token.unsqueeze(0).unsqueeze(0).expand(batch_size_hm, B, -1).to(conditional.device)

            heatmap_input = torch.cat([heatmap_x_embedding, heatmap_y_embedding, intensity_emb, heatmap_token], axis=2)

        denoising_step_emb = self.denoising_step_emb(time)  # batch_size x 3*dim_embedding
        denoising_step_emb = denoising_step_emb.unsqueeze(1)  # batch_size x 1 x 3*dim_embedding
        
        denoising_step_token = self.denoising_step_token.reshape(1, 1, self.dim_embedding)  # 1 x 1 x dim_embedding
        denoising_step_token = denoising_step_token.expand(batch_size, 1, self.dim_embedding)  # batch_size x 1 x dim_embedding
        denoising_step_token = denoising_step_token.to(noisy_traj.device)

        denoising_step_input = torch.cat([denoising_step_emb, denoising_step_token], axis=2)   # batch_size x 1 x 4*dim_embedding

        # ic(heatmap_x_embedding.shape,
        #    heatmap_y_embedding.shape,
        #    intensity_emb.shape,
        #    heatmap_token.shape,
        #    heatmap_input.shape)
        if conditional is None:
            model_input = torch.cat([noisy_trajectory_input, denoising_step_input], axis=1).float()
        else:
            model_input = torch.cat([noisy_trajectory_input, heatmap_input, denoising_step_input], axis=1).float()

        # ic(model_input.shape)

        output = self.encoder(model_input)
        # Decode the part of the representation corresponding to the (denoised) noisy trajectory
        denoised_trajectory = self.trajectory_decoder(output[:, :self.seq_len, :])
        denoised_trajectory = denoised_trajectory.transpose(1, 2)

        if predict_conditional:
            assert conditional is not None
            heatmap_embedding = output[:, self.seq_len:self.seq_len+576, :].transpose(1, 2)
            heatmap_embedding = heatmap_embedding.reshape(batch_size, -1, 24, 24)
            reconstructed_heatmap = self.heatmap_decoder(heatmap_embedding)
            return denoised_trajectory, reconstructed_heatmap
        else:
            return denoised_trajectory
