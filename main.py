import argparse
import torch
import torch_geometric

import os
from tqdm import tqdm
import numpy as np
import time
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from models import Gradid
from utils import set_seed, format_time, CosineAnnealingWarmUpRestarts, EarlyStopping

def make_subgraph(embedding, label):
    subgraph_list = []
    for each_body, each_label in zip(tqdm(embedding), torch.from_numpy(label)):
        leng = len(each_body)
        x_ = torch.tensor(each_body)
        edge_index = torch_geometric.utils.grid(leng, leng, dtype=torch.long)[1][:leng].t().contiguous()
        subgraph = torch_geometric.data.Data(x=x_, edge_index=edge_index, y=each_label)
        subgraph_list.append(subgraph)
    return subgraph_list

def construct_data(args):
    data_info = args.data_info
    
    if args.only_train:
        split_list = ['train', 'valid']
    elif args.only_test:
        split_list = ['test']
    else:
        split_list = ['train', 'valid', 'test']

    try:
        return_graph = []
        for each_split in split_list:
            sentence_embedding = np.load(f"{args.embedding_path}/{data_info}_{each_split}_{args.embedding_model}.npy", allow_pickle=True)
            label = np.load(f'data/{data_info}/{each_split}_label.npy')
            return_graph.append(make_subgraph(sentence_embedding, label))
        return return_graph

    except FileNotFoundError:
        print("Cannot find saved embeddings. Please run 'python sentence_embedding.py' with needed sentence model & data")

def train(model, train_graph, valid_graph, args):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=len(train_graph)//args.batch_size, T_mult=2, eta_max=args.lr, T_up=10, gamma=0.8)
    early_stopping = EarlyStopping(patience=args.earlystop_patience, verbose=True, path=args.save_path)
    optimizer.zero_grad()
    epoch = 1
    while True:
        t0 = time.time()
        train_loss, valid_loss, valid_accuracy, valid_f1, valid_auroc = 0.0, 0.0, 0.0, 0.0, 0.0
        epoch_loss = 0.0
        model.train()
		
        for graph in DataLoader(train_graph, batch_size=args.batch_size, shuffle=True): # num_workers=os.cpu_count()//2-1
            optimizer.zero_grad()
            graph = graph.to(args.device)
            out = model(graph)
            loss = loss_fn(out, graph.y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            scheduler.step()

        train_loss = float(epoch_loss / len(train_graph))
        valid_loss, valid_accuracy, valid_f1, valid_auroc = evaluate(model, valid_graph, True, args.device, args)
        print(f"EPOCH: {epoch}  ||  Elapsed: {format_time(time.time()-t0)}.")
        print(f"   Train_loss: {train_loss:.4f} | Valid_loss: {valid_loss:.4f}  ||  Valid_acc: {valid_accuracy:.4f} | Valid_F1: {valid_f1:.4f} | Valid_auroc: {valid_auroc:.4f}")
        early_stopping(valid_loss, model)
        print("")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        epoch += 1


def evaluate(model, eval_graph, is_valid, device, args):
    if is_valid:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        epoch_loss = 0.0
    else:
        model.load_state_dict(torch.load(args.load_path if args.load_path else args.save_path))
    eval_real_list, eval_pred_list, eval_score_list = [], [], []
    with torch.no_grad():
        model.eval()
        for graph in tqdm(DataLoader(eval_graph)):
            graph = graph.to(args.device)
            out = model(graph)
            if is_valid:
                loss = loss_fn(out.unsqueeze(dim=0), graph.y)
                epoch_loss += loss.item()
            eval_real_list.append(graph.y.detach().cpu())
            eval_pred_list.append(out.argmax().unsqueeze(dim=0).detach().cpu())
            eval_score_list.append(torch.sigmoid(out[1]).unsqueeze(dim=0).detach().cpu())
        eval_real_list = torch.cat(eval_real_list)
        eval_pred_list = torch.cat(eval_pred_list)
        eval_score_list = torch.cat(eval_score_list)

        eval_accuracy = accuracy_score(eval_real_list, eval_pred_list)
        eval_f1 = f1_score(eval_real_list, eval_pred_list, average='macro')
        #eval_precision = precision_score(eval_real_list, eval_pred_list, pos_label=0)
        #eval_recall = precision_score(eval_real_list, eval_pred_list, pos_label=0)
        eval_auroc = roc_auc_score(eval_real_list, eval_score_list)
        if is_valid:
            return float(epoch_loss / len(eval_graph)), eval_accuracy, eval_f1, eval_auroc
        else:
            print()
            print(f'  Test Acc: {eval_accuracy:.4f}, Test F1: {eval_f1:.4f}, Test AUROC: {eval_auroc:.4f}')
            
            #print(f'  Test Acc: {(eval_accuracy*100):.2f}%, Test Precision: {(eval_precision*100):.2f}%, Test Recall: {(eval_recall*100):.2f}%')
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_info", default='nela')

    parser.add_argument("--embedding_model", default='all-roberta-large-v1')
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=5e-6, type=float)
    parser.add_argument("--embedding_path", default='save_embedding/')
    parser.add_argument("--save_path", default='save_checkpoint/')
    
    parser.add_argument("--earlystop_patience", default=10, type=int)
    parser.add_argument("--rand_seed", default=None, type=int)
    parser.add_argument("--only_train", action='store_true')
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--load_path", default=None)

#    parser.add_argument("--in_channels", type=int)
#	parser.add_argument("--inter_channels", default=2048, type=int)
    parser.add_argument("--mlp_dim", default=512, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    
    parser.add_argument("--dropout_p", default=0.5, type=float)
    parser.add_argument("--num_heads_1", default=4, type=int)
    parser.add_argument("--num_heads_2", default=2, type=int)
    

    
    args = parser.parse_args()
    args.embedding_model = args.embedding_model.split('/')[-1]

    #if args.rand_seed: # In experiements, we set seeds to [111, 222, 333, 444, 555]
    set_seed(args.rand_seed)
    print("[Load data...]")
    graph_list = construct_data(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.in_channels = graph_list[0][0].num_features
    args.inter_channels = 2 * args.in_channels//args.num_heads_1
    if args.save_path is 'save_checkpoint/':
        args.save_path = f'save_checkpoint/{args.embedding_model}_{args.lr}_{args.data_info}.pt'
    model = Gradid(args).to(args.device)

    if not args.only_test:
        train(model, graph_list[0], graph_list[1], args) # Train & Validation: train_graph, valid_graph
    if not args.only_train:
        print()
        print('[Start test!!!]')
        evaluate(model, graph_list[-1], False, args.device, args) # Test: test_graph

if __name__ == "__main__":
    main()