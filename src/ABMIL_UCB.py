import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
from torch import einsum
from einops import rearrange, repeat

class ABMIL_UCB(nn.Module):
    """
    Multi-headed attention network with optional gating, integrating UCB for patch selection.
    Uses tanh-attention and sigmoid-gating as in ABMIL (https://arxiv.org/abs/1802.04712).
    The UCB mechanism is applied to attention scores to prioritize patches based on their
    Upper Confidence Bound, balancing exploration and exploitation.

    Args:
        feature_dim (int): Input feature dimension.
        head_dim (int): Hidden layer dimension for each attention head. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.
        n_branches (int): Number of attention branches. Defaults to 1, but can be set to
                          n_classes to generate one set of attention scores for each class.
        gated (bool): If True, sigmoid gating is applied. Otherwise, the simple attention
                      mechanism is used.
    """

    def __init__(self, feature_dim=1024, head_dim=256, n_heads=8, dropout=0.2, n_branches=1, gated=False):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads
        self.n_branches = n_branches # Store n_branches for UCB_Score

        # Initialize attention head(s)
        self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                               nn.Tanh(),
                                                               nn.Dropout(dropout)) for _ in range(n_heads)])

        # Initialize gating layers if gating is used
        if self.gated:
            self.gating_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                                   nn.Sigmoid(),
                                                                   nn.Dropout(dropout)) for _ in range(n_heads)])

        # Initialize branching layers
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, n_branches) for _ in range(n_heads)])

        # Initialize condensing layer if multiple heads are used
        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)

    def UCB_Score(self, attention_scores, beta, current_iter, count_score, num_images, top_k=10):
        """
        Calculates UCB scores for patches and selects top-k patches.

        Args:
            attention_scores (torch.Tensor): Raw attention scores for a single head and branch.
                                             Shape: (batch_size, num_images, n_branches).
            beta (float): Exploration parameter for UCB.
            current_iter (int): Current training iteration/counter.
            count_score (torch.Tensor): Accumulated counts of patch selections.
                                        Shape: (batch_size, num_images, n_branches).
            num_images (int): Total number of images/patches.
            top_k (int): Number of top patches to select based on UCB score.

        Returns:
            tuple: A tuple containing:
                - new_attention_scores (torch.Tensor): Attention scores after UCB-based selection and re-normalization.
                                                       Shape: (batch_size, num_images, n_branches).
                - updated_count_score (torch.Tensor): Updated accumulated patch selection counts.
                                                      Shape: (batch_size, num_images, n_branches).
        """
        # Ensure count_score is not zero for log calculation in UCB formula
        # Add a small epsilon to avoid log(0) and division by zero for score_sum
        score_sum = torch.sum(count_score, dim=1, keepdim=True) + 1e-6 # Sum over num_images dimension
        
        # Ensure current_iter is at least 1 for log calculation
        log_iter = torch.tensor(np.log(max(1, current_iter)), dtype=torch.float32).to(attention_scores.device)

        # Calculate UCB score: Attention_Score + beta * sqrt(log(iter) / sum_of_counts)
        # UCB_score shape: (batch_size, num_images, n_branches)
        UCB_score = attention_scores + beta * torch.sqrt(log_iter / score_sum)

        # Select top-k patches based on UCB score for each batch and branch
        # max_indices shape: (batch_size, top_k, n_branches)
        _, max_indices = torch.topk(UCB_score, k=top_k, dim=1)

        # Create one-hot vectors from selected indices
        # one_hot_vector shape: (batch_size, top_k, n_branches, num_images)
        one_hot_vector = F.one_hot(max_indices, num_classes=num_images).float()
        
        # Sum along the top_k dimension to get a mask for selected patches
        # summed_mask shape: (batch_size, n_branches, num_images)
        summed_mask = one_hot_vector.sum(dim=1)
        
        # Permute to match attention_scores shape: (batch_size, num_images, n_branches)
        summed_mask = summed_mask.permute(0, 2, 1)

        # Update the count_score for selected patches
        updated_count_score = count_score + summed_mask

        # Apply the mask to the original attention scores
        new_attention_scores = attention_scores * summed_mask
        
        # Re-normalize the selected attention scores. Add epsilon to prevent division by zero.
        normalization_factor = torch.sum(new_attention_scores, dim=1, keepdim=True) + 1e-6
        new_attention_scores = new_attention_scores / normalization_factor
        
        return new_attention_scores, updated_count_score

    def forward(self, features, attn_mask=None, counter=0, UCB_Count_Score=None, ucb=False, ucb_beta=1.0, ucb_warmup_iter=500, ucb_top_k=10):
        """
        Forward pass with optional UCB-based patch selection.

        Args:
            features (torch.Tensor): Input features, acting as queries and values.
                                     Shape: (batch_size x num_images x feature_dim).
            attn_mask (torch.Tensor): Attention mask to enforce zero attention on empty images.
                                      Defaults to None. Shape: (batch_size x num_images).
            counter (int): Current training iteration, used for UCB warm-up. Defaults to 0.
            UCB_Count_Score (torch.Tensor, optional): Accumulated counts of patch selections
                                                       across batches, images, branches, and heads.
                                                       Shape: (batch_size, num_images, n_branches, n_heads).
                                                       If None, it will be initialized to zeros.
            ucb (bool): If True, UCB mechanism is activated after warm-up. Defaults to False.
            ucb_beta (float): Exploration parameter for UCB. Defaults to 1.0.
            ucb_warmup_iter (int): Number of iterations before UCB is activated. Defaults to 500.
            ucb_top_k (int): Number of top patches to select based on UCB score. Defaults to 10.

        Returns:
            tuple: A tuple containing:
                - aggregated_features (torch.Tensor): Attention-weighted features aggregated across heads.
                                                      Shape: (batch_size x n_branches x feature_dim).
                - head_attentions (torch.Tensor): Raw attention scores from each head (before softmax or UCB).
                                                  Shape: (batch_size x n_branches x n_heads x num_images).
                - UCB_Count_Score (torch.Tensor): Updated accumulated patch selection counts.
                                                  Shape: (batch_size, num_images, n_branches, n_heads).
        """

        assert features.dim() == 3, f'Input features must be 3-dimensional (batch_size x num_images x feature_dim). Got {features.shape} instead.'
        if attn_mask is not None:
            assert attn_mask.dim() == 2, f'Attention mask must be 2-dimensional (batch_size x num_images). Got {attn_mask.shape} instead.'
            assert features.shape[:2] == attn_mask.shape, f'Batch size and number of images must match between features and mask. Got {features.shape[:2]} and {attn_mask.shape} instead.'

        batch_size, num_images, feature_dim = features.shape

        # Initialize UCB_Count_Score if not provided
        if UCB_Count_Score is None:
            # Shape: (batch_size, num_images, n_branches, n_heads)
            UCB_Count_Score = torch.zeros(batch_size, num_images, self.n_branches, self.n_heads,
                                          device=features.device, dtype=torch.float32)

        head_attentions = [] # Stores raw attention scores for visualization/debugging
        head_features = []   # Stores weighted features from each head

        for i in range(self.n_heads): # Iterate through each attention head
            attention_vectors = self.attention_heads[i](features)        # Main attention vectors (shape: batch_size x num_images x head_dim)

            if self.gated:
                gating_vectors = self.gating_layers[i](features)                # Gating vectors (shape: batch_size x num_images x head_dim)
                attention_vectors = attention_vectors.mul(gating_vectors)       # Element-wise multiplication to apply gating vectors

            # Raw attention scores for each branch (shape: batch_size x num_images x n_branches)
            attention_scores = self.branching_layers[i](attention_vectors)

            # Set attention scores for empty images to -inf (before UCB or softmax)
            if attn_mask is not None:
                # Mask is automatically broadcasted to shape: batch_size x num_images x n_branches
                attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(-1), -1e9)

            # Get the count score slice for the current head
            current_head_ucb_count = UCB_Count_Score[:, :, :, i] # Shape: (batch_size, num_images, n_branches)

            if ucb and counter > ucb_warmup_iter:
                # Apply UCB to attention scores
                attention_scores_modified, updated_head_ucb_count = self.UCB_Score(
                    attention_scores=attention_scores.clone(), # Clone to avoid modifying original attention_scores
                    beta=ucb_beta,
                    current_iter=counter,
                    count_score=current_head_ucb_count,
                    num_images=num_images,
                    top_k=ucb_top_k
                )
                # Update the global UCB_Count_Score with the updated counts for this head
                UCB_Count_Score[:, :, :, i] = updated_head_ucb_count
                
                # Apply softmax to the UCB-modified attention scores
                attention_scores_softmax = F.softmax(attention_scores_modified, dim=1)
            else:
                # If UCB is not active (e.g., during warm-up or if ucb=False), use standard softmax
                attention_scores_softmax = F.softmax(attention_scores, dim=1)

            # Multiply features by attention scores to get weighted features for this head
            # weighted_features shape: (batch_size x n_branches x feature_dim)
            weighted_features = torch.einsum('bnr,bnf->brf', attention_scores_softmax, features)

            head_attentions.append(attention_scores) # Store the original attention scores (before UCB/softmax for this head)
            head_features.append(weighted_features)

        # Concatenate multi-head outputs along the feature dimension
        # aggregated_features shape: (batch_size x n_branches x (n_heads * feature_dim))
        aggregated_features = torch.cat(head_features, dim=-1)
        if self.n_heads > 1:
            # Condense features if multiple heads are used
            # aggregated_features shape: (batch_size x n_branches x feature_dim)
            aggregated_features = self.condensing_layer(aggregated_features)

        # Stack raw attention scores from all heads for potential analysis/visualization
        # head_attentions list contains (batch_size x num_images x n_branches) for each head
        # Stacked shape: (batch_size x num_images x n_branches x n_heads)
        head_attentions = torch.stack(head_attentions, dim=-1)
        # Rearrange to (batch_size x n_branches x n_heads x num_images) for consistency if needed
        head_attentions = rearrange(head_attentions, 'b n r h -> b r h n')

        return aggregated_features, head_attentions, UCB_Count_Score

