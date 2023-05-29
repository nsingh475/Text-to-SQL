import pandas as pd
import itertools

class Prepare_Raw_Data():

    """Prepares Raw Dataset for Text-to-SQL task"""
    
    def __init__(self, in_path, out_path, wiki_file, db_file, out_file):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.wiki_file = wiki_file
        self.db_file = db_file
        self.out_file = out_file
        
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def make_case_insensitive(col_value, col_type):
            if col_type == "NLQ": ## col_value will be string
                return col_value.lower()
            elif col_type == "SQL": ## col_value will be a dictionary
                conds = col_value['conds']
                for cond in conds:
                    cond[-1] = str(cond[-1]).lower()
                return col_value
            elif col_type == "Column": ## col_value will be a list
                for i in range(len(col_value)):
                    col_value[i] = col_value[i].lower()
                return col_value
            
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## Reading files    
        wiki = pd.read_json(self.in_path+self.wiki_file, lines=True)
       
        database = pd.read_json(self.in_path+self.db_file, lines=True)
        database.reset_index(drop=True, inplace=True)
        database.rename({'id': 'table_id'}, axis=1, inplace=True) ## renaming id column in database so that the wikiSQL and database table has same column name
        
        ## Pre-processing
        raw=pd.merge(wiki,database, on='table_id')
        raw = raw.drop(columns=['phase', 'table_id', 'types', 'rows', 'name', 'page_title', 'section_title', 'caption', 'page_id'])
        
        raw['NL_query'] = raw.apply(lambda row: make_case_insensitive(row.question, "NLQ"), axis=1)
        raw['GroundTruth_SQL'] = raw.apply(lambda row: make_case_insensitive(row.sql, "SQL"), axis=1)
        raw['Table_Headers'] = raw.apply(lambda row: make_case_insensitive(row.header, "Column"), axis=1)
        raw = raw.drop(columns=['question', 'sql', 'header'])
        
        ## writing files
        raw.to_csv(self.out_path+self.out_file)
        
        return raw
        

class Generate_BIO_labels():

    """Prepares training data with BIO labels for NER """
    
    def __init__(self, df, out_path, out_file_list):
        super().__init__()
        self.bio_df = df
        self.out_path = out_path
        self.out_file_list = out_file_list
        
    def run(self):
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def generate_BIO_labels(nlq, conds):
            nlq = nlq.lower()
            if nlq[-1] in ['.', '?', '!', ':', ';', ',']:  ## other punct - ['#', '"', '(', ')', '/', '€', '$', "'", '%']: 
                    nlq = nlq[:-1]
            nlq_token = nlq.split()
            nlq_len = len(nlq_token)
            bio_label = ['O']*nlq_len

            for cond in conds:
                value = str(cond[-1]).lower()
                vals_token = value.split()
                len_vals = len(vals_token)

                for pos, token in enumerate(nlq_token):
                    if vals_token[0] in token:
                        start_ind = pos
                        bio_label[start_ind] = 'B'
                        break

                if len_vals>0:
                    new_nlq = nlq_token[start_ind:]
                    for pos, token in enumerate(new_nlq):
                        if str(vals_token[-1]) in str(token):
                            end_ind = pos+start_ind
                            for i in range(start_ind+1, end_ind+1):
                                bio_label[i] = 'I' 
                            break
            return bio_label
        
        def convert_to_SpaCy_format(nlq, bio):
            if nlq[-1] in ['.', '?', '!']:
                nlq = nlq[:-1]
            nlq_tok = nlq.split()

            def get_ind(word, nlq):
                start_ind = nlq.index(word)
                end_ind = start_ind + len(word) #-1
                return start_ind, end_ind


            value_list = []
            for pos, label in enumerate(bio):
                if label == 'B':
                    word = nlq_tok[pos]
                    start_ind, end_index = get_ind(word, nlq)
                    value_list.append((start_ind, end_index, label))
                elif label == 'I':
                    word = nlq_tok[pos]
                    shift = end_index + 1
                    start_ind, end_index = get_ind(word, nlq[shift:])

            return (nlq, {'entities': value_list})    
        
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        self.bio_df['BIO_label'] = self.bio_df.apply(lambda row: generate_BIO_labels(row.NL_query.strip(), row.GroundTruth_SQL['conds']), axis=1)
        self.bio_df = self.bio_df.drop(columns=['Table_Headers'])
        
        bio_df_spacy = self.bio_df.copy()
        bio_df_spacy['SpaCy_input'] = bio_df_spacy.apply(lambda row: convert_to_SpaCy_format(row.NL_query, row.BIO_label), axis=1)
        
        
        ## writing files
        self.bio_df.to_csv(self.out_path+self.out_file_list[0])
        bio_df_spacy.to_csv(self.out_path+self.out_file_list[1])
        
        return bio_df_spacy


class Transform_df_in_SQuAD_format():

    """Transform training data into SQuAD format for BERT based MRC"""
    
    def __init__(self, df, out_path, out_file_list):
        super().__init__()
        self.context_df = df
        self.out_path = out_path
        self.out_file_list = out_file_list
        
    def run(self):
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def create_context(nlq, headers):
            if nlq[-1] in ['.', '?', '!', ':', ';', ',']: # remove the punctuation at the end of NLQ
                nlq = nlq[:-1]
            header_string = ' '.join(headers)
            agg_functions = 'Aggregate empty minimum maximum count sum average'
            return 'Query '+nlq+' Headers '+header_string+' '+agg_functions
        
        def template_for_filter_col_questions(cond_list):
            filter_qs = []
            for cond in cond_list:
                value = cond[2]
                filter_qs.append(f'what is the filter column for {value}')
            return filter_qs  
        
        def generate_questions(sql):
            cond_list = sql['conds']
            sel_q     = 'what is the select column'
            agg_q     = 'what is the aggregate function'
            val_q     = 'what are the values'
            filter_qs = template_for_filter_col_questions(cond_list)
            return {'select_column': sel_q, 'aggregate_function': agg_q, 'values': val_q, 'filter_columns': filter_qs}
        
        def get_answer(key, sql, headers, context, isFilter=False):
            context = context
            if key =='sel':
                k = headers[int(sql[key])]
            elif key == 'agg':
                k = ['empty', 'maximum', 'minimum', 'count', 'sum', 'average'][int(sql[key])]
            elif isFilter==True:
                k = headers[key]
            else:
                k = key
            k = str(k)
            start = context.find(k)
            end = context.find(k) + len(k)
            return {'text': k, 'start': start, 'end': end}
        
        def generate_answer(sql, headers, context): 
            ans_dict = {}
            ## get answer for select and aggregate
            ans = get_answer('sel', sql, headers, context) #key = 'sel' key of sql dictionary
            ans_dict['select_column'] = ans
            ans = get_answer('agg', sql, headers, context) #key = 'agg' key of sql dictionary
            ans_dict['aggregate_function'] = ans

            cond_list = sql['conds']
            ## get answer for value
            val_list = []
            for cond in cond_list:
                ans = get_answer(cond[2], sql, headers, context) #key = value of where condition
                val_list.append(ans)
            ans_dict['values'] = val_list

            ## get answer for filter column
            filter_col_list = []
            for cond in cond_list:
                ans = get_answer(int(cond[0]), sql, headers, context, True) # key = column number
                filter_col_list.append(ans)
            ans_dict['filter_columns'] = filter_col_list
            return ans_dict
        
        def generate_squad_triplet(row):
        # def generate_squad_triplet(sql, header, context):
            sql = row.GroundTruth_SQL
            header = row.Table_Headers
            context = row.Context

            # Generate template questions
            ques = generate_questions(sql)
            # Generate answers for each question
            ans = generate_answer(sql, header, context)

            ## Generating the <Q,A,C> triplet
            triplet = []
            for key in ques.keys():
                if key == 'values':
                    q = ques[key]
                    a = ans[key]
                    single_value_flag = 0   
                    triplet.append([q,a,context,single_value_flag])
                elif key == 'filter_columns':
                    for pos in range(len(ques[key])):
                        q = ques[key][pos]
                        a = ans[key][pos]
                        single_value_flag = 1 
                        triplet.append([q,a,context,single_value_flag])       
                else:
                    q = ques[key]
                    a = ans[key]
                    single_value_flag = 1
                    triplet.append([q,a,context, single_value_flag])       
            return triplet
        
        

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## creating context
        self.context_df['Context'] = self.context_df.apply(lambda row: create_context(row.NL_query.strip(), row.Table_Headers), axis=1)
        
        ## Generating SQuAD triplet
        squad_input = self.context_df.copy()
        squad_input['triplet_list'] = squad_input.apply (lambda row: generate_squad_triplet(row), axis=1)
        
        triplet_list = squad_input['triplet_list'].values
        triplet_list_flat = list(itertools.chain(*triplet_list))
        
        SQuAD_format_df = pd.DataFrame(triplet_list_flat, columns = ['Question', 'Answer', 'Context', 'Single_Value_Flag'])
            
        SQuAD_single_val = SQuAD_format_df.loc[SQuAD_format_df['Single_Value_Flag'] == 1]
        SQuAD_single_val = SQuAD_single_val.drop(columns=['Single_Value_Flag'])
        
        SQuAD_multi_val = SQuAD_format_df.loc[SQuAD_format_df['Single_Value_Flag'] == 0]
        SQuAD_multi_val = SQuAD_multi_val.drop(columns=['Single_Value_Flag'])
        
        ## writing files
        self.context_df.to_csv(self.out_path+self.out_file_list[0])
        SQuAD_format_df.to_csv(self.out_path+self.out_file_list[1])
        SQuAD_single_val.to_csv(self.out_path+self.out_file_list[2])
        SQuAD_multi_val.to_csv(self.out_path+self.out_file_list[3])
        
        return  SQuAD_single_val, SQuAD_multi_val