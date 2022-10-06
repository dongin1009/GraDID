import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import re
import os
import argparse
from nltk.tokenize import sent_tokenize

from sklearn.model_selection import train_test_split

def remove_stopword(df, seq_level):
	df.body = df.body.replace('[^A-Za-z0-9.?!,\'|<EOS>|<EOP> ]+', '', regex=True)
	df.body = df.body.replace(' \.', '.', regex=True)
	df.body = df.body.replace(' \!', '!', regex=True)
	df.body = df.body.replace(' \?', '?', regex=True)
	df.body = df.body.replace(' \,', ',', regex=True)
	df.body = df.body.replace(' +', ' ', regex=True)
	if seq_level == 'paragraph' or 'p':
		df.body = df.body.replace(" <EOS>", '')
		df.body = df.body.str.split(" <EOP> ")
	elif seq_level == 'sentence' or 's':
		df.body = df.body.str[:-7].str.split(" <EOS> <EOP> | <EOS> ")
	else:
		print('Sequence type error !')
	df.body = df.body.replace(' \'', '\'', regex=True)
	df.body = df.body.replace('\' ', '\'', regex=True)
	return df

def sentence_embedding(sentence_model, doc, device):
	embedding = []
	for article in tqdm(doc):
		temp_tensor = sentence_model.encode(article, device=device)
		embedding.append(temp_tensor)
	return np.array(embedding, dtype=object)

def data_selection(data_info, seq_level='paragraph'):
	if data_info=='nela':
		column_names = ['ind', 'headline', 'body', 'label']
		train_df = pd.read_csv('data/nela/train.csv', names=column_names)
		valid_df = pd.read_csv('data/nela/dev.csv', names=column_names)
		test_df = pd.read_csv('data/nela/test.csv', names=column_names)
		train_df, valid_df, test_df = remove_stopword(train_df, seq_level), remove_stopword(valid_df, seq_level), remove_stopword(test_df, seq_level)
		return [train_df, valid_df, test_df]

	elif data_info=='yh':
		df = pd.read_csv('data/yh/yh_dataset.csv')
		df.body = df.body.apply(sent_tokenize)
		train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
		valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=42, stratify=valid_df.label)
		df_reset = lambda x: x.reset_index(drop=True)
		train_df, valid_df, test_df = df_reset(train_df), df_reset(valid_df), df_reset(test_df)
		return [train_df, valid_df, test_df]

	else:
		print("Unsupported data")


def main():	
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_info", default='nela', choices=['nela', 'yh'], type=str)
	parser.add_argument("--output_path", default='save_embedding/', type=str)
	parser.add_argument("--sentence_model", default='all-roberta-large-v1', type=str)
	parser.add_argument("--seq_level", default='paragraph', type=str)

	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SentenceTransformer(args.sentence_model)

	df_list = data_selection(args.data_info, args.seq_level)
	split_list = ['train', 'valid', 'test']

	for each_df, each_split in zip(df_list, split_list):
		if not os.path.isfile(f'data/{args.data_info}/*_label.npy'):
			np.save(f'data/{args.data_info}/{each_split}_label.npy', np.array(each_df.label))

		embedding_path = f"{args.output_path}/{args.data_info}_{each_split}_{args.sentence_model.split('/')[-1]}.npy"
		if not os.path.isfile(embedding_path):
			each_embedding = sentence_embedding(model, each_df.body.values, device)
			np.save(embedding_path, each_embedding)

if __name__=='__main__':
	main()