import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GnnLayer(torch.nn.Module):
	def __init__(self, args):
		super(GnnLayer, self).__init__()
		self.conv1 = GATConv(args.in_channels, args.inter_channels, heads=args.num_heads_1, add_self_loops=False)
		#self.conv1 = GATConv(in_channels, in_channels//2, heads=4, add_self_loops=False)
		self.conv2 = GATConv(args.inter_channels * args.num_heads_1, args.in_channels//args.num_heads_2, heads=args.num_heads_2, add_self_loops=False)
		#self.conv2 = GATConv(in_channels*2, in_channels//2, heads=2, add_self_loops=False)
		self.device = args.device

	def forward(self, graph):
		x, edge_index = graph.x, graph.edge_index
		x = self.conv1(x, edge_index) # if want to check attention, add argument 'return_attention_weights=True')
		x = F.gelu(x)
		x = self.conv2(x, edge_index)
		if self.training: # for mini-batch training, get the last node.
			super_node = x[torch.cat((graph.batch.diff()==1, torch.tensor([True], device=x.device)))]
		else:
			super_node = x[-1]
		return super_node

class ClassifiLayer(torch.nn.Module):
	def __init__(self, args):
		super(ClassifiLayer, self).__init__()
		self.linear1 = torch.nn.Linear(args.in_channels, args.mlp_dim)
		self.linear2 = torch.nn.Linear(args.mlp_dim, args.num_classes)
		self.layernorm1 = torch.nn.LayerNorm((args.in_channels,), eps=1e-10, elementwise_affine=True)
		self.layernorm2 = torch.nn.LayerNorm((args.mlp_dim,), eps=1e-10, elementwise_affine=True)
		self.dropout_p = args.dropout_p

	def forward(self, node):
		node = F.gelu(self.layernorm1(node))
		node = F.dropout(node, p=self.dropout_p, training=self.training)
		node = self.linear1(node)
		node = F.gelu(self.layernorm2(node))
		inconsist_type = self.linear2(node)
		return inconsist_type

class Gradid(torch.nn.Module):
	def __init__(self, args):
		super(Gradid, self).__init__()
		self.gnn = GnnLayer(args)
		self.classify = ClassifiLayer(args)
	
	def forward(self, graph):
		super_node = self.gnn(graph)
		result = self.classify(super_node)
		return result
