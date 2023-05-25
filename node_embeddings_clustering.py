import torch

import networkx as nx

from models import GNN_MODELS
from prepare_data import get_graph_data_loader
from classification import get_classification_predictions
from clustering import clustering_pipeline


def graph_clustering(graph, model_name='sage', with_permutations=False):
    if not isinstance(graph, nx.Graph):
        raise TypeError(f'Параметр `graph` не является объектом networkx.Graph')
    
    if model_name not in GNN_MODELS:
        raise KeyError(f'Выбрана неизвестная модель `{model_name}`. Список доступных моделей: {", ".join(list(GNN_MODELS.keys()))}')
    
    dataset_loader = get_graph_data_loader(graph, with_permutations)

    model = GNN_MODELS[model_name]()
    model.load_state_dict(torch.load(f'model_states/{model_name}'))

   
    nodes_classification = get_classification_predictions(model, dataset_loader)
    pred_clusters = clustering_pipeline(graph, nodes_classification.numpy(), with_permutations)

    return pred_clusters