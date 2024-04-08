import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, top_k, num_hidden):
        super().__init__()
        self.linear = nn.Linear(num_hidden, 1)
        self.top_k = top_k

    def forward(self, x):
        #get scores for each token
        scores = self.linear(x)
        #get top k tokens 
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=1)
        #get topk from x
        x_top_k = x.gather(1, top_k_indices.expand(-1, -1, x.size(-1)))

        return x_top_k, top_k_indices