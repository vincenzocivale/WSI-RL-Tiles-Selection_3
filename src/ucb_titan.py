import torch
import torch.nn as nn
import math

# Ensure preprocess_features is accessible. If running as a standalone script,
# copy the preprocess_features function from the original TITAN vision_transformer.py here.
# Example:
# from your_titan_library.vision_transformer import preprocess_features

def preprocess_features(features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int):
    """
    Preprocesses raw patch features and coordinates into a grid format.
    Adapted from MahmoodLab/TITAN vision_transformer.py.
    """
    features = features.squeeze(0) if features.dim() == 3 else features
    coords = coords.squeeze(0) if coords.dim() == 3 else coords

    offset = coords.min(dim=0).values
    grid_coords = torch.floor_divide(coords - offset, patch_size_lv0)

    grid_offset = grid_coords.min(dim=0).values
    grid_coords = grid_coords - grid_offset
    _H, _W = grid_coords.max(dim=0).values + 1

    feature_grid = torch.zeros((_H, _W, features.size(-1)), device=features.device, dtype=features.dtype)
    coords_grid = torch.zeros((_H, _W, 2), dtype=torch.int64, device=coords.device)

    indices = grid_coords[:, 0] * _W + grid_coords[:, 1]
    feature_grid.view(-1, features.size(-1)).index_add_(0, indices, features)
    coords_grid.view(-1, 2).index_add_(0, indices, coords)

    feature_grid = feature_grid.permute(2, 0, 1)
    coords_grid = coords_grid.permute(2, 0, 1)

    bg_mask = torch.any(feature_grid != 0, dim=0)
    return feature_grid.unsqueeze(0), coords_grid.unsqueeze(0), bg_mask.unsqueeze(0)


class UCBAdaptiveTitan(nn.Module):
    """
    A wrapper around the pre-trained MahmoodLab/TITAN model to integrate
    Upper Confidence Bound (UCB) patch selection into its Vision Transformer's
    forward pass.
    """
    def __init__(self, original_titan_model, top_k_patches_ratio=0.2, ucb_c_constant=math.sqrt(2)):
        super().__init__()
        self.original_titan_model = original_titan_model
        self.original_titan_model.eval() # Ensure original model is in evaluation mode

        # Extract relevant sub-modules from the pre-trained TITAN model's vision_encoder
        self.vision_encoder = self.original_titan_model.vision_encoder
        self.patch_embed = self.vision_encoder.patch_embed
        self.cls_token = self.vision_encoder.cls_token
        self.pos_drop = self.vision_encoder.pos_drop
        self.norm_pre = self.vision_encoder.norm_pre
        self.blocks_module_list = self.vision_encoder.blocks.modules_list
        self.norm = self.vision_encoder.norm
        self.fc_norm = self.vision_encoder.fc_norm
        self.head = self.vision_encoder.head
        self.attn_pool = self.vision_encoder.attn_pool
        self.attn_pool_contrastive = self.vision_encoder.attn_pool_contrastive
        self.proj = getattr(self.vision_encoder, 'proj', None)

        # UCB specific parameters
        self.top_k_patches_ratio = top_k_patches_ratio
        self.ucb_c_constant = ucb_c_constant
        self.num_prefix_tokens = self.vision_encoder.num_prefix_tokens

        # Methods from original VisionTransformer (assuming they are accessible)
        self._pos_embed_method = self.vision_encoder._pos_embed
        self.get_alibi_method = self.vision_encoder.get_alibi
        self.pos_encode_type = self.vision_encoder.pos_encode_type
        self.attn_pool_type = getattr(self.vision_encoder, 'attn_pool_type', None)
        self.return_all_tokens = self.vision_encoder.return_all_tokens

    def forward(self, patch_features, patch_coords, patch_size_lv0, **kwargs):
        patch_features = patch_features.to(self.patch_embed[0].weight.dtype)
        
        # Add batch dimension if not present
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
            patch_coords = patch_coords.unsqueeze(0)

        # Preprocess features into a grid format
        x_grid, coords_grid, bg_mask_processed = preprocess_features(patch_features, patch_coords, patch_size_lv0)

        B, C, H, W = x_grid.shape
        x = x_grid.flatten(2, 3).transpose(1, 2) # (B, H*W, C)
        num_grid_elements = H * W

        # Apply initial patch embedding
        x = self.patch_embed(x)

        # Add class token if exists
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Apply positional embedding
        x = self._pos_embed_method(x, coords_grid, W, H)
        x = self.norm_pre(x)

        # Handle background mask for batch_size=1 (filters x directly)
        if bg_mask_processed is not None and B == 1:
            bg_mask_flat_for_x = torch.cat((torch.ones((1, self.num_prefix_tokens), dtype=torch.bool, device=x.device), bg_mask_processed.view(1, -1)), dim=1)
            x = x[bg_mask_flat_for_x].unsqueeze(0)
        
        # Initialize UCB statistics for the current slide/batch
        current_sequence_length = x.shape[1]
        ucb_total_reward = torch.zeros(B, current_sequence_length, device=x.device, dtype=x.dtype)
        ucb_count_score = torch.zeros(B, current_sequence_length, device=x.device, dtype=x.dtype)

        # Ensure class token is always considered selected
        if self.num_prefix_tokens > 0:
            ucb_total_reward[:, :self.num_prefix_tokens] = float('inf')
            ucb_count_score[:, :self.num_prefix_tokens] = 1.0 # Start count at 1 for log term

        # Active patches mask based on initial preprocessing
        active_patches_mask = bg_mask_processed.view(B, -1) if bg_mask_processed is not None else torch.ones(B, num_grid_elements, dtype=torch.bool, device=x.device)
        if self.num_prefix_tokens > 0:
            cls_token_active_mask = torch.ones((B, self.num_prefix_tokens), dtype=torch.bool, device=x.device)
            active_patches_mask = torch.cat((cls_token_active_mask, active_patches_mask), dim=1)


        # Iteration counter for UCB's log term
        ucb_iteration_counter = 1

        # Iterate through Transformer blocks with UCB selection
        for level, block in enumerate(self.blocks_module_list):
            # Generate Alibi attention bias
            attn_bias = None
            if self.pos_encode_type == 'alibi':
                attn_bias = self.get_alibi_method(W, H, bg_mask_processed if B == 1 else None)
                attn_bias = attn_bias.repeat(B, 1, 1, 1).type(x.dtype).to(x.device)

            # Calculate rewards for current patches (excluding class token)
            current_patch_embeddings = x[:, self.num_prefix_tokens:, :]
            current_rewards = torch.norm(current_patch_embeddings, dim=-1) # (B, num_patches)

            # Update UCB statistics for active patches
            ucb_total_reward[:, self.num_prefix_tokens:][active_patches_mask[:, self.num_prefix_tokens:]] += current_rewards[active_patches_mask[:, self.num_prefix_tokens:]]
            ucb_count_score[:, self.num_prefix_tokens:][active_patches_mask[:, self.num_prefix_tokens:]] += 1
            ucb_iteration_counter += 1

            # Calculate UCB scores
            mean_rewards = torch.where(ucb_count_score > 0, ucb_total_reward / ucb_count_score, torch.tensor(0.0, device=x.device, dtype=x.dtype))
            total_pulls_for_level = ucb_count_score.sum(dim=1, keepdim=True)
            exploration_term = torch.where(ucb_count_score > 0,
                                           self.ucb_c_constant * torch.sqrt(
                                               torch.log(total_pulls_for_level + 1e-6) / (ucb_count_score + 1e-6)
                                           ),
                                           torch.tensor(float('inf'), device=x.device, dtype=x.dtype))
            ucb_scores = mean_rewards + exploration_term

            # Mask out non-active patches for UCB selection
            ucb_scores = torch.where(active_patches_mask, ucb_scores, torch.tensor(float('-inf'), device=x.device, dtype=x.dtype))
            
            # Ensure class token is always selected
            if self.num_prefix_tokens > 0:
                ucb_scores[:, :self.num_prefix_tokens] = float('inf')

            # Select top K promising patches
            top_k_for_this_level = max(1, int(num_grid_elements * self.top_k_patches_ratio))
            _, selected_indices = torch.topk(ucb_scores, k=min(top_k_for_this_level + self.num_prefix_tokens, current_sequence_length), dim=-1)
            
            # Create a new attention mask for the current block
            level_attention_mask = torch.zeros_like(active_patches_mask, dtype=torch.bool)
            level_attention_mask.scatter_(1, selected_indices, True)
            final_attention_mask_for_block = active_patches_mask & level_attention_mask

            # Pass x and attention mask to the Transformer block
            x = block(x, attn_bias=attn_bias, bg_mask=final_attention_mask_for_block)

            # Update active_patches_mask for the next level
            active_patches_mask = final_attention_mask_for_block.clone()

        # Final Normalization and Pooling
        x = self.norm(x)

        # Apply final pooling using AttentionalPooler
        if self.attn_pool is not None:
            pooled_tokens = self.attn_pool(x, bg_mask=active_patches_mask)
            if self.attn_pool_type == 'parallel':
                pooled = self.attn_pool_contrastive(x, bg_mask=active_patches_mask)
            else: # 'cascade'
                pooled = self.attn_pool_contrastive(pooled_tokens, bg_mask=None)
            
            pooled = pooled.squeeze(1)

            if kwargs.get('no_proj', False):
                return pooled
            
            if self.proj is not None:
                pooled = pooled @ self.proj

            if self.return_all_tokens:
                # Filter x to return only selected tokens from the last level
                filtered_x = x[active_patches_mask].unsqueeze(0) if B == 1 else x # Simplified for B=1
                return pooled, filtered_x
            else:
                return pooled
        else: # iBOT style pooling (mean or class token)
            if self.return_all_tokens:
                filtered_x = x[active_patches_mask].unsqueeze(0) if B == 1 else x
                return filtered_x
            
            if self.num_prefix_tokens > 0:
                pooled = x[:, self.num_prefix_tokens:][active_patches_mask[:, self.num_prefix_tokens:]].mean(dim=0).unsqueeze(0)
            else:
                pooled = x[active_patches_mask].mean(dim=0).unsqueeze(0)
            
            pooled = self.fc_norm(pooled)
            pooled = self.head(pooled)
            return pooled