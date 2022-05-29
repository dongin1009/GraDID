import argparse
import torch
import torch_geometric

import os
from tqdm import tqdm
import numpy as np
import time
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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

def construct_data(embedding_path, data_info, sentence_model):
    train_sentence_embedding = np.load(f"{embedding_path}/{data_info}_train_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)
    valid_sentence_embedding = np.load(f"{embedding_path}/{data_info}_valid_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)
    test_sentence_embedding = np.load(f"{embedding_path}/{data_info}_test_{sentence_model.split('/')[-1]}.npy", allow_pickle=True)

    train_label = np.load(f'data/{data_info}/train_label.npy')
    valid_label = np.load(f'data/{data_info}/valid_label.npy')
    test_label = np.load(f'data/{data_info}/test_label.npy')
    
    train_graph = make_subgraph(train_sentence_embedding, train_label)
    valid_graph = make_subgraph(valid_sentence_embedding, valid_label)
    test_graph = make_subgraph(test_sentence_embedding, test_label)
    
    return train_graph, valid_graph, test_graph

def train(model, train_graph, valid_graph, test_graph, args):
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
            # wandb.log({"Train Loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
        train_loss = float(epoch_loss / len(train_graph))
        valid_loss, valid_accuracy, valid_f1, valid_auroc = evaluate(model, valid_graph, True, args)
        print(f"EPOCH: {epoch}  ||  Elapsed: {format_time(time.time()-t0)}.")
        print(f"   Train_loss: {train_loss:.4f} | Valid_loss: {valid_loss:.4f}  ||  Valid_acc: {valid_accuracy:.4f} | Valid_F1: {valid_f1:.4f} | Valid_auroc: {valid_auroc:.4f}")
        early_stopping(valid_loss, model)
        print("")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        epoch += 1

    evaluate(model, test_graph, False, args)
		
def evaluate(model, eval_graph, is_valid, device, args):
    if is_valid:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        epoch_loss = 0.0
    else:
        model.load_state_dict(torch.load(args.save_path))
    eval_real_list, eval_pred_list, eval_score_list = [], [], []
    with torch.no_grad():
        model.eval()
        for graph in DataLoader(eval_graph):
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
        eval_auroc = roc_auc_score(eval_real_list, eval_score_list)
        if is_valid:
            return float(epoch_loss / len(eval_graph)), eval_accuracy, eval_f1, eval_auroc
        else:
            print(f'  Test_acc: {eval_accuracy:.4f}, Test_f1: {eval_f1:.4f}, Test_auroc: {eval_auroc:.4f}')
            

def main():
	parser = argparse.ArgumentParser()
#	parser.add_argument("--in_channels", default=1024, type=int)
#	parser.add_argument("--inter_channels", default=2048, type=int)
	parser.add_argument("--mlp_dim", default=512, type=int)
	parser.add_argument("--num_classes", default=2, type=int)

	parser.add_argument("--dropout_p", default=0.5, type=float)
	parser.add_argument("--num_heads_1", default=4, type=int)
	parser.add_argument("--num_heads_2", default=2, type=int)

	parser.add_argument("--data_info", default='nela')
	parser.add_argument("--embedding_path", default='save_embedding/')
	parser.add_argument("--embedding_model", default='all-roberta-large-v1')
#	parser.add_argument("--save_path", default='save_checkpoint/best_model.pt')	
	parser.add_argument("--epochs", default=100, type=int)
	parser.add_argument("--batch_size", default=64, type=int)
	parser.add_argument("--lr", default=0.0001, type=float)
	parser.add_argument("--weight_decay", default=5e-6, type=float)

	parser.add_argument("--earlystop_patience", default=10, type=int)
	parser.add_argument("--rand_seed", default=None, type=int)
	
    args = parser.parse_args()
    if args.rand_seed: # In experiemnts, we set seeds to [111, 222, 333, 444, 555]
        set_seed(args.rand_seed)

	train_graph, valid_graph, test_graph = construct_data(args.embedding_path, args.data_info, args.embedding_model)
	
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.in_channels = train_graph[0].num_features
	args.inter_channels = 2 * args.in_channels//args.num_heads_1
	args.save_path = f'save_checkpoint/{args.embedding_model}_{args.lr}_{args.data_info}.pt'
	model = Gratid(args).to(args.device)

	train(model, train_graph, valid_graph, test_graph, args)

if __name__ == "__main__":
    main()