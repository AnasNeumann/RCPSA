import torch
from torch.nn import Module, Linear, Sequential, ReLU, ModuleList, LayerNorm, Dropout
import torch.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch_geometric.data import HeteroData
from conf import O, P, D, R, S

# ==================================================
# =*= Model file for the HGAT model architecture =*=
# ==================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

EMBEDDING_DIMENSION: int = 12
ATTENTION_HEADS: int     = 4
GNN_STACK_SIZE: int      = 2
DROPOUT_RATE: float      = 0.1

class HyperGraphGNN(Module):
    def __init__(self, task_features:int, resource_features:int, demand_features:int, d_model:int=EMBEDDING_DIMENSION, num_heads:int=ATTENTION_HEADS, num_layers:int=GNN_STACK_SIZE):
        super(HyperGraphGNN, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.task_expanded = Linear(task_features, d_model)
        self.resource_expanded = Linear(resource_features, d_model)
        self.demande_expanded = Linear(demand_features, d_model)
        self.GAT_tasks_for_resources = ModuleList()
        self.GAT_tasks_preds = ModuleList()
        self.GAT_tasks_succs = ModuleList()
        self.GAT_resources_for_tasks = ModuleList()
        for _ in range(num_layers):
            self.GAT_tasks_for_resources.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False))
            self.GAT_tasks_preds.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False))
            self.GAT_tasks_succs.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False))
            self.GAT_resources_for_tasks.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False))
        self.aggregation_mlp_tasks = Sequential(
            Linear(4 * d_model, d_model), ReLU(),
            LayerNorm(d_model), Dropout(DROPOUT_RATE))
        self.aggregation_mlp_resources = Sequential(
            Linear(2 * d_model, d_model), ReLU(),
            LayerNorm(d_model), Dropout(DROPOUT_RATE))
        self.task_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.resource_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.q_value_head = Sequential(
            Linear(3 * d_model, d_model), ReLU(),
            Linear(d_model, d_model // 2), ReLU(),
            Linear(d_model // 2, 1))

    def forward(self, data: HeteroData):
        x_tasks      = self.task_expanded(data[O].x)                   # [num_tasks, d_model]
        x_resources  = self.resource_expanded(data[R].x)               # [num_resources, d_model]
        demande_attr = self.demande_expanded(data[O, D, R].edge_attr)  # [num_edges, d_model]
        for i in range(self.num_layers):
            x_resources_with_tasks = self.GAT_tasks_for_resources[i]((x_tasks, x_resources), data[O, D, R].edge_index, edge_attr=demande_attr)
            x_resources            = self.aggregation_mlp_resources(torch.cat([x_resources, x_resources_with_tasks], dim=-1))
            x_tasks_with_preds     = self.GAT_tasks_preds[i]((x_tasks, x_tasks),data[O, P, O].edge_index)
            x_tasks_with_succs     = self.GAT_tasks_succs[i]((x_tasks, x_tasks),data[O, S, O].edge_index)
            x_tasks_with_resources = self.GAT_resources_for_tasks[i]((x_resources, x_tasks),data[R, D, O].edge_index)
            x_tasks                = self.aggregation_mlp_tasks(torch.cat([x_tasks, x_tasks_with_preds, x_tasks_with_succs, x_tasks_with_resources], dim=-1))
        pooled_tasks          = self.task_pooling(x_tasks, data[O].batch)            # [batch_size, d_model]
        pooled_resources      = self.resource_pooling(x_resources, data[R].batch)    # [batch_size, d_model]     
        state_vector          = torch.cat([pooled_tasks, pooled_resources], dim=-1)  # [batch_size, 2 * d_model]
        state_vector_expanded = state_vector[data[O].batch]                          # [num_tasks, 2 * d_model]
        inputs                = torch.cat([x_tasks, state_vector_expanded], dim=1)   # [num_tasks, 3 * d_model] in a batch settings: num_tasks = num_total_tasks_across_batch
        q_values              = self.q_value_head(inputs)                            # [num_tasks, 1]
        return q_values