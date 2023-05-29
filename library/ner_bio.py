from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import pandas as pd
import ast 
import pickle


class Train_BIO_NER():
    
    """Train model to learn to generate BIO labels for each token in NLQ"""
    
    def __init__(self, in_path, file_name, labels, model, new_model_name, out_path, n_iter, load_mdl_flag):
        super().__init__()
        self.in_path = in_path 
        self.file_name = file_name
        self.labels = labels 
        self.model = model 
        self.new_model_name = new_model_name
        self.out_path = out_path 
        self.n_iter = n_iter
        self.load_mdl_flag = load_mdl_flag

        
    def run(self):
        
        ## ------------------------------------- Helper functions ----------------------------------------------------------## 
        def save_model(mdl, out_path, new_model_name):
            if out_path is not None:
                output_dir = Path(self.out_path+'NER/')
                if not output_dir.exists():
                    output_dir.mkdir()
                mdl.meta['name'] = new_model_name  # rename model
                mdl.to_disk(output_dir)
                print("Saved model to", output_dir)

        ## ------------------------------------- Main Execution ------------------------------------------------------------##

        train_df = pd.read_csv(self.in_path+self.file_name)
        train_list = list(train_df['SpaCy_input'].values)
        train_data = [ast.literal_eval(i) for i in train_list]
        
        if self.load_mdl_flag == 'True':
            print('Loading a pre-saved model')
            mdl = spacy.load(self.out_path+'NER/')
        elif self.model is not None:
            print('Loading model')
            mdl = spacy.load(self.model)  # load existing spacy model
            print("Loaded model '%s'" % self.model)
        else:
            print('Creating a new model')
            mdl = spacy.blank('en')  # create blank Language class
            print("Created blank 'en' model")

        if 'ner' not in mdl.pipe_names:
            ner = mdl.create_pipe('ner')
            mdl.add_pipe('ner') # mdl.add_pipe(ner)  ### use this in case of spacy v2
        else:
            ner = mdl.get_pipe('ner')

        for i in self.labels :
            ner.add_label(i)   # Add new entity labels to entity recognizer

        if self.model is None:
            optimizer = mdl.begin_training()
        else:
            optimizer = mdl.entity.create_optimizer()

        # Get names of other pipes to disable them during training to train only NER
        other_pipes = [pipe for pipe in mdl.pipe_names if pipe != 'ner']
        with mdl.disable_pipes(*other_pipes):  # only train NER
            for itn in range(self.n_iter):
                print(f'------ Iteration {itn+1} ------')
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=compounding(4., 32., 1.001))
                for batch in batches: 
                    """ Below commented code works for Spacy v2
                    texts, annotations = zip(*batch)
                    mdl.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)"""
                    for text, annotations in batch:
                        doc = mdl.make_doc(text)
                        try:
                            example = Example.from_dict(doc, annotations) # create Example
                        except:
                            print('skipping')
                        mdl.update([example], sgd=optimizer, losses=losses, drop=0.35) # Update the model
                    
                print('Losses', losses)
                if (itn+1)%5 == 0: ## saving intermediate model after 5 iterations
                    print('Saving Intermediate model')
                    save_model(mdl, self.out_path, self.new_model_name)

        # Save model 
        print('Saving Final model')
        save_model(mdl, self.out_path, self.new_model_name)
        return mdl
        
class Predict_BIO_NER():
    
    """Train model to learn to generate BIO labels for each token in NLQ"""
    
    def __init__(self, in_path, file_name, model_path, model_name):
        super().__init__()
        self.in_path = in_path
        self.file_name = file_name
        self.model_path = model_path 
        self.model_name = model_name 
        
    def run(self):
        
        ## ------------------------------------- Helper functions ----------------------------------------------------------## 
        def write_data_to_pickle(data, file_name, path):
            with open(path+file_name, 'wb') as f:
                pickle.dump(data, f)

        def convert_format(input):
            temp = input.strip(')(').split(', {')
            temp[1] = "{"+temp[1] 
            res = [ast.literal_eval(i) for i in temp]
            out = temp[1]
            result = ast.literal_eval(out)
            output = (temp[0][1:-1], result)
            return output
                
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        # load BIO file
        print('Reading Input File...')
        bio_df = pd.read_csv(self.in_path+self.file_name)
        bio_df['GroundTruth'] = bio_df.apply(lambda row: convert_format(row.SpaCy_input), axis=1)

        # load model
        print('Loading NER Model...')
        nlp = spacy.load(self.model_path+self.model_name)


        text_list = bio_df['NL_query'].values
        ground_truth = bio_df['GroundTruth'].values # [(NLQ, {'entities': [(start, end, label), ()]})]

        # predicting
        print('Predicting...')
        eval_data = []
        for txt in text_list:
            ent_list = []
            doc = nlp(txt)
            for ent in doc.ents:
                ent_list.append((ent.label_, ent.text))    
            eval_data.append((txt, ent_list)) # [(B, text), (I, text), ...]
            
        ## extracting GT answers
        ans_data = []
        for val in ground_truth:
            nlq = val[0]
            entities = val[1]['entities']
            ans_list = []
            for ans in entities:
                ans_list.append((ans[2], nlq[ans[0]:ans[1]]))
            ans_data.append(ans_list)
                
        ## NER prediction Data
        eval_data_gt_ans = zip(eval_data, ans_data)
        final_ans = []
        for i in eval_data_gt_ans:
            final_ans.append([i[0][0], i[0][1], i[1]])
            
        ## create dataframe and write data as pickle
        print('Writing prediction file...')
        df = pd.DataFrame(final_ans, columns=['nlq', 'prediction', 'ground_truth'])
        write_data_to_pickle(df, 'NER_predictions.pickle', self.in_path)

        return df


class Evaluate_NER():
    
    """Evaluate NER model on test/ dev set"""
    
    def __init__(self, df):
        super().__init__()
        self.df = df 
        
    def run(self):
        
        ## ------------------------------------- Helper functions ----------------------------------------------------------## 
        def accuracy(pred_gt):
            num_vals = []
            accuracy = []
            for pred, gt in pred_gt:
                if len(pred) == len (gt):
                    val = 1
                    if (len(pred) == 0): 
                        acc = 1
                    elif (len(pred) == 1):
                        if gt[0][1][:-1] in pred[0][1]: acc = 1
                        else: acc = 0
                    else:
                        for i in range(len(pred)):
                            if gt[i][1][:-1] not in pred[i][1]:
                                acc = 0
                                break
                            else:
                                acc = 1
                else:
                    val = 0
                    acc = 0
                accuracy.append(acc)
                num_vals.append(val)

            acc_len = sum(num_vals)*100/len(num_vals)
            acc_pred = sum(accuracy)*100/len(accuracy)
            return  round(acc_len, 2),  round(acc_pred, 2)
                
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        pred = list(self.df['prediction'].values)
        gt =  list(self.df['ground_truth'].values)
        pred_gt = zip(pred, gt)
        acc_len, acc_pred = accuracy(pred_gt)

        return acc_len, acc_pred

