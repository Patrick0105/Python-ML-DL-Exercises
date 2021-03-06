# -*- coding: utf-8 -*-
"""Taipei_QA_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13PFtgfg2jXKPBdQHilwyWbZEjgcA-X9J
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
dir = '/content/gdrive/MyDrive/'
filename = 'Taipei_QA_new.txt'

df = pd.read_csv(dir + filename,sep = ' ',error_bad_lines = False , header = None)
df.columns = ['department','question']
df

!pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

question_list = df['question'].tolist()
question_tokens_list= tokenizer.__call__(question_list).input_ids
question_tokens_list[0]

vocab = tokenizer.vocab
print(question_tokens_list[0])
pairs = [(word,idx) for word, idx in vocab.items() if idx in question_tokens_list[0]]
print(pairs)

print(pd.Categorical(df['department']))
cat_num = pd.Categorical(df['department']).codes.max() + 1
df['cat'] = pd.Categorical(df['department'])
df['cat'] = df['cat'].cat.codes

df

from sklearn.model_selection import train_test_split
import numpy as np

training_questions,testing_questions,training_tags,testing_tags = train_test_split(np.array(question_tokens_list),np.array(df['cat'])),train_size = 0.7

print(training_tags)