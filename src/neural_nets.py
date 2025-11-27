import torch
from torch.nn import Module, Linear, Sequential, ReLU, ModuleList, LayerNorm, Dropout, Embedding
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch_geometric.data import HeteroData

from conf import O, P, D, R, S, EMBEDDING_DIMENSION, ATTENTION_HEADS, GNN_STACK_SIZE, DROPOUT_RATE, TASK_ID_DIM, RESOURCE_ID_DIM

# ==================================================
# =*= Model file for the HGAT model architecture =*=
# ==================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class HyperGraphGNN(Module):
    def __init__(self, task_features:int, resource_features:int, demand_features:int, num_tasks: int, num_resources: int, d_model:int=EMBEDDING_DIMENSION, num_heads:int=ATTENTION_HEADS, num_layers:int=GNN_STACK_SIZE, attn_dropout: float=DROPOUT_RATE, task_id_dim: int = TASK_ID_DIM, resource_id_dim: int = RESOURCE_ID_DIM):
        super(HyperGraphGNN, self).__init__()
        self.num_layers            = num_layers
        self.d_model               = d_model
        self.task_id_dim           = task_id_dim
        self.resource_id_dim       = resource_id_dim
        self.num_tasks             = num_tasks
        self.num_resources         = num_resources
        self.task_id_embedding     = Embedding(num_tasks, task_id_dim)
        self.resource_id_embedding = Embedding(num_resources, resource_id_dim)
        self.task_expanded         = Linear(task_features + task_id_dim, d_model)
        self.resource_expanded     = Linear(resource_features + resource_id_dim, d_model)
        self.demande_expanded      = Linear(demand_features, d_model)
        self.GAT_tasks_for_resources   = ModuleList()
        self.GAT_tasks_preds           = ModuleList()
        self.GAT_tasks_succs           = ModuleList()
        self.GAT_resources_for_tasks   = ModuleList()
        self.aggregation_mlp_tasks     = ModuleList()
        self.aggregation_mlp_resources = ModuleList()
        for _ in range(num_layers):
            self.GAT_tasks_for_resources.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False, dropout=attn_dropout, edge_dim=d_model))
            self.GAT_tasks_preds.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False, dropout=attn_dropout))
            self.GAT_tasks_succs.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False, dropout=attn_dropout))
            self.GAT_resources_for_tasks.append(GATConv(in_channels=(d_model, d_model), out_channels=d_model, heads=num_heads, concat=False, dropout=attn_dropout, edge_dim=d_model))
            self.aggregation_mlp_tasks.append(Sequential(
                Linear(4 * d_model, d_model), ReLU(),
                LayerNorm(d_model), Dropout(attn_dropout)))
            self.aggregation_mlp_resources.append(Sequential(
                Linear(2 * d_model, d_model), ReLU(),
                LayerNorm(d_model), Dropout(attn_dropout)))
        self.task_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.resource_pooling = AttentionalAggregation(Linear(d_model, 1))
        self.q_value_head = Sequential(
            Linear(3 * d_model, d_model), ReLU(),
            Linear(d_model, d_model // 2), ReLU(),
            Linear(d_model // 2, 1))

    @staticmethod
    def _build_local_ids(batch_index: torch.Tensor) -> tuple[torch.Tensor, int]:
        counts    = torch.bincount(batch_index)
        prefix    = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
        local_ids = torch.arange(batch_index.size(0), device=batch_index.device, dtype=torch.long) - prefix[batch_index]
        return local_ids

    def forward(self, data: HeteroData):
        task_ids             = self._build_local_ids(data[O].batch)
        resource_ids         = self._build_local_ids(data[R].batch)
        task_id_features     = self.task_id_embedding(task_ids)
        resource_id_features = self.resource_id_embedding(resource_ids)
        task_inputs          = torch.cat([data[O].x, task_id_features], dim=-1)
        resource_inputs      = torch.cat([data[R].x, resource_id_features], dim=-1)
        x_tasks              = self.task_expanded(task_inputs)                 # [num_tasks, d_model]
        x_resources          = self.resource_expanded(resource_inputs)         # [num_resources, d_model]
        demande_attr         = self.demande_expanded(data[O, D, R].edge_attr)  # [num_edges, d_model]
        for i in range(self.num_layers):
            x_resources_with_tasks = self.GAT_tasks_for_resources[i]((x_tasks, x_resources), data[O, D, R].edge_index, edge_attr=demande_attr)
            x_resources            = self.aggregation_mlp_resources[i](torch.cat([x_resources, x_resources_with_tasks], dim=-1))
            x_tasks_with_preds     = self.GAT_tasks_preds[i]((x_tasks, x_tasks), data[O, P, O].edge_index)
            x_tasks_with_succs     = self.GAT_tasks_succs[i]((x_tasks, x_tasks), data[O, S, O].edge_index)
            x_tasks_with_resources = self.GAT_resources_for_tasks[i]((x_resources, x_tasks), data[R, D, O].edge_index, edge_attr=demande_attr)
            x_tasks                = self.aggregation_mlp_tasks[i](torch.cat([x_tasks, x_tasks_with_preds, x_tasks_with_succs, x_tasks_with_resources], dim=-1))
        pooled_tasks          = self.task_pooling(x_tasks, data[O].batch)            # [batch_size, d_model]
        pooled_resources      = self.resource_pooling(x_resources, data[R].batch)    # [batch_size, d_model]     
        state_vector          = torch.cat([pooled_tasks, pooled_resources], dim=-1)  # [batch_size, 2 * d_model]
        state_vector_expanded = state_vector[data[O].batch]                          # [num_tasks, 2 * d_model]
        inputs                = torch.cat([x_tasks, state_vector_expanded], dim=1)   # [num_tasks, 3 * d_model] in a batch settings: num_tasks = num_total_tasks_across_batch
        q_values              = self.q_value_head(inputs)                            # [num_tasks, 1]
        return q_values
