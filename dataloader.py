from tqdm import tqdm
import torch
import numpy as np
import torch_geometric

def make_subgraph(embedding, label):
    subgraph_list = []
    for each_body, each_label in zip(tqdm(embedding), torch.from_numpy(label)):
        leng = len(each_body)
        x_ = torch.tensor(each_body)
        edge_index = torch_geometric.utils.grid(leng, leng, dtype=torch.long)[1][:leng].t().contiguous()
        subgraph = torch_geometric.data.Data(x=x_, edge_index=edge_index, y=each_label)
        subgraph_list.append(subgraph)
    return subgraph_list

def construct_data(output_path, data_info, sentence_model):
    train_sentence_embedding = np.load(f"{output_path}/{data_info}_train_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)
    valid_sentence_embedding = np.load(f"{output_path}/{data_info}_valid_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)
    test_sentence_embedding = np.load(f"{output_path}/{data_info}_test_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)

    train_label = np.load('data/{data_info}/train_label.npy')
    valid_label = np.load('data/{data_info}/valid_label.npy')
    test_label = np.load('data/{data_info}/test_label.npy')
    
    train_graph = make_subgraph(train_sentence_embedding, train_label)
    valid_graph = make_subgraph(valid_sentence_embedding, valid_label)
    test_graph = make_subgraph(test_sentence_embedding, test_label)
    return train_graph, valid_graph, test_graph