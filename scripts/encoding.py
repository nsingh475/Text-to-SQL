import sys
mode = sys.argv[1] ## getting argument from commandline

from library.preprocess import *
from library.mrc_bert import *
from library.ner_bio import *

in_path = f'Data/Input/{mode}/'
out_path = f'Data/Output/{mode}/'
enc_path = out_path + 'batch_encoding/'

## Prepare Raw data 
wiki_file = f'{mode}.jsonl'
db_file = f'{mode}.tables.jsonl'
out_file = f'WikiSQL_processed.csv'
data_obj = Prepare_Raw_Data(in_path, out_path, wiki_file, db_file, out_file)
raw = data_obj.run()

## Generate BIO labels for training NER
df = raw.copy()
out_file = [f'BIO_labels.csv', f'BIO_labels_SpaCy.csv']
bio_label_obj = Generate_BIO_labels(df, out_path, out_file)
bio_df_spacy = bio_label_obj.run()

## Transforming data into SQuAD format to train BERT based MRC
df = raw.copy()
out_file_list = ['WikiSQL_with_context.csv', 'WikiSQL_in_squad_format.csv', 'WikiSQL_squad_format_single_val.csv', 'WikiSQL_squad_format_multi_val.csv']
squad_obj = Transform_df_in_SQuAD_format(df, out_path, out_file_list)
SQuAD_single_val, SQuAD_multi_val = squad_obj.run()

## Generating BERT encodings
df = SQuAD_single_val.copy()
batch_size = 512
tokenizer_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
file_name = 'batch_encoding'

if mode == 'train':
    chunk_size = 100
else:
    chunk_size = -1  ## do not chunk test and dev data
BERT_enc_obj = Generating_BERT_encodings(df, batch_size, tokenizer_name, file_name, enc_path, chunk_size)
batch_encoding = BERT_enc_obj.run()