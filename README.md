# GraDID: Detecting Documents with Inconsistent Context
This repository contains official code of **Detecting Documents with Inconsistent Context**. This model, named GraDID, is a Graph-based Document Inconsistency Detection model that avoids suffering from bad quality texts. Misinformation and clickbait are easy to spread and writing, so we propose automatic content inconsistency check model. This model regard to contents by sequence-level(sentence or paragraph) and each sequence turn into a node which used in graph structure.

To process this model, follow this step.

### Get the Data
1. Download [nela-17](https://github.com/BenjaminDHorne/NELA2017-Dataset-v1)
2. To make incongruent content of news, refer to [previous preprocessing method](https://github.com/sugoiii/detecting-incongruity-dataset-gen).
### Data Preprocessing
Extract sentence-level embedding of news and save embedding as numpy array. Use pretrained `sentence-transformers` via [sbert.net](https://www.sbert.net/docs/pretrained_models.html). To get more models, explore [huggingface](https://huggingface.co/sentence-transformers).
##### Nela-17
```python
python sentence_embedding.py --sentence_model='all-roberta-large-v1'
```
##### YH
```python
python sentence_embedding.py --data_info='yh' --sentence_model='Huffon/sentence-klue-roberta-base' 
```

### Train
##### Nela-17
```python
python main.py --embedding_model='all-roberta-large-v1'
```
##### YH
```python
python main.py --data_info='yh' --embedding_model='Huffon/sentence-klue-roberta-base'