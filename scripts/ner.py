import sys
mode = sys.argv[1] ## getting argument from commandline

try:
    load_mdl_flag = sys.argv[2]
except:
    load_mdl_flag = 'False' ## Flag to enable loading of a model from local

from library.ner_bio import *

in_path = f'Data/Input/{mode}/'
out_path = f'Data/Output/{mode}/'
model_path = f'Data/model/'

file_name = 'BIO_labels_SpaCy.csv'


if mode == 'train': ## Training NER
    labels = ['B', 'O', 'I']
    new_model_name='BIO_label'
    n_iter=12
    model = None
    BIO_obj = Train_BIO_NER(out_path, file_name, labels, model, new_model_name, model_path, n_iter, load_mdl_flag)
    out_model = BIO_obj.run()
else: ## evaluating / testing NER
    model_name = 'NER'
    NER_Obj = Predict_BIO_NER(out_path, file_name, model_path, model_name)
    df = NER_Obj.run() 

    Eval_NER_Obj = Evaluate_NER(df)
    acc_len, acc_pred = Eval_NER_Obj.run() 

    print('---------------------------------------------- NER Evaluation Report ----------------------------------------------')
    print(f'Accuracy in predicting number of values in where clause: {acc_len}')
    print(f'NER Accuracy: {acc_pred}')
    