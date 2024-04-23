import warnings
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


class TriangularConstraint(_Loss):
    
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']

    def __init__(self, margin=0.0,  eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction='mean'):
        super(TriangularConstraint, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.eps = eps
        self.swap = swap

    def forward(self, node_costs,nodeInsDel,edge_costs,edgeInsDel):
        node_loss=torch.max(node_costs/(2.0*nodeInsDel+self.margin),torch.ones_like(node_costs))        
        node_upper_part=torch.triu_indices(node_loss.shape[0],node_loss.shape[1],offset=1)
        
        if self.reduction=='mean':
            final_node_loss=node_loss[node_upper_part[0],node_upper_part[1]].mean()
        else:
            final_node_loss=node_loss[node_upper_part[0],node_upper_part[1]].sum()

        
        if edge_costs.shape[0]==0:
            return final_node_loss
        
        edge_loss=torch.max(edge_costs/(2.0*edgeInsDel+self.margin),torch.ones_like(edge_costs))
        edge_upper_part=torch.triu_indices(edge_loss.shape[0],edge_loss.shape[1],offset=1)
        
        if self.reduction=='mean':
            return .5*(final_node_loss+edge_loss[edge_upper_part[0],edge_upper_part[1]].mean())

        return .5*(final_node_loss+edge_loss[edge_upper_part[0],edge_upper_part[1]].sum())


class ReducedTriangularConstraint(_Loss):
    
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']

    def __init__(self, margin=0.0,  eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction='mean'):
        super(ReducedTriangularConstraint, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.eps = eps
        self.swap = swap

    def forward(self, node_sub,nodeInsDel,edge_costs,edgeInsDel):
        node_loss=torch.max(node_sub-2*nodeInsDel+self.margin,torch.zeros(1,device=node_sub.device))        
        
        
        if edge_costs.shape[0]==0:
            return node_loss
        
        edge_loss=torch.max(edge_costs-2*edgeInsDel+self.margin,torch.zeros_like(edge_costs))
        edge_upper_part=torch.triu_indices(edge_loss.shape[0],edge_loss.shape[1],offset=1)
        
        if self.reduction=='mean':
            return node_loss+edge_loss[edge_upper_part[0],edge_upper_part[1]].mean()
        
        return node_loss+edge_loss[edge_upper_part[0],edge_upper_part[1]].sum()
