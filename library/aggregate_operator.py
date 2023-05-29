import ast
import pandas as pd 


class Predict_AggregateFunction_and_Operator():

    """Predict aggregate_function and where_clause_operators for dev/ test data"""
    
    def __init__(self, rule_path, out_path, agg_file, op_file, wiki_file):
        super().__init__()
        self.rule_path = rule_path
        self.out_path = out_path
        self.agg_file = agg_file
        self.op_file = op_file
        self.wiki_file = wiki_file
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def tokenize(nlq):
            if nlq[-1] in ['.', '?', '!']:
                nlq = nlq[:-1]
            nlq_tok = nlq.split()
            clean_nlq_tok = clean_tokens(nlq_tok)
            return clean_nlq_tok

        def clean_tokens(nlq_tok):
            det = ['a', 'an', 'the']
            prep = ['in', 'for', 'of', 'at', 'by', 'from', 'with', 'on']
            be_form = ['is', 'am', 'are', 'were', 'was', 'being', 'be', 'been']
            have_form = ['have', 'had', 'has']
            do_form = ['do', 'does', 'did']
            exta_words = ['me', 'tell', 'if', 'that', 'and', 'as', 'you', 'name', 'there']
            remove_words = det+prep+be_form+have_form+do_form+exta_words
            nlq_tokens_copy = nlq_tok.copy()
            for token in nlq_tokens_copy:
                if token in remove_words:
                    nlq_tok.remove(token)
            return nlq_tok

        def get_gt_agg_func(sql):
            agg_func = ['empty', 'max',  'min', 'count', 'sum', 'avg']
            gt_sql = ast.literal_eval(sql)
            agg_func_ind = gt_sql['agg']
            return agg_func[int(agg_func_ind)]


        def get_gt_operators(sql):
            op_list = ['=', '>', '<']
            gt_sql = ast.literal_eval(sql)
            where_conds = gt_sql['conds']
            operators = []
            for cond in where_conds:
                op = cond[1]
                if int(op) == 3: op = 0
                operators.append(op_list[int(op)])
            return operators

        def prepare_rules(pickle_file):
            word_key_dict = {}
            for key, values in pickle_file.items():
                for word in values:
                    word_key_dict[word]=key
            return word_key_dict

        def predict_agg(tok_nlq, rules_dict):
            for tok in tok_nlq:
                if tok in rules_dict.keys():
                    return  rules_dict[tok]
            return 'empty'              
            
        def predict_op(tok_nlq, num, rules_dict):
            op_list = []
            for tok in tok_nlq:
                if tok in rules_dict.keys():
                    op_list.append(rules_dict[tok])
            pad = ['=']*(num - len(op_list))
            op_list += pad
            return op_list

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## Reading files
        print('Reading Input file...')
        wiki = pd.read_csv(self.out_path+self.wiki_file)
        agg_pkl_file = pd.read_pickle(self.rule_path+self.agg_file)
        op_pkl_file = pd.read_pickle(self.rule_path+self.op_file)
       
        ## tokenizing nlq
        wiki['tok_nlq'] = wiki.apply(lambda row: tokenize(row.NL_query.strip()), axis=1) 
        wiki['gt_agg'] = wiki.apply(lambda row: get_gt_agg_func(row.GroundTruth_SQL.strip()), axis=1)
        wiki['gt_op'] = wiki.apply(lambda row: get_gt_operators(row.GroundTruth_SQL.strip()), axis=1)

        ## prepare rules
        print('Loading Rule files...')
        agg_rules_dict = prepare_rules(agg_pkl_file)
        op_rules_dict = prepare_rules(op_pkl_file)

        ## predict aggregate function and operator
        print('Predicting...')
        wiki['pred_agg'] =  wiki.apply(lambda row: predict_agg(row.tok_nlq, agg_rules_dict), axis=1)
        wiki['pred_op'] =  wiki.apply(lambda row: predict_op(row.tok_nlq, len(row.gt_op), op_rules_dict), axis=1)     
        
        ## writing files
        print('Writing File...')
        wiki.to_csv(self.out_path+'Aggregate_Operator_prediction.csv')
        
        return wiki


class Evaluate_AggregateFunction_and_Operator():
    
    """Evaluate accuracy in predicting aggregate function and operator on test/ dev set"""
    
    def __init__(self, df):
        super().__init__()
        self.df = df 
        
    def run(self):
        
        ## ------------------------------------- Helper functions ----------------------------------------------------------## 
        def get_agg_func_accuracy(gt_pred_agg):
            acc = []
            for gt, pred in gt_pred_agg:
                if pred == gt: acc.append(1)
                else: acc.append(0)
            return sum(acc)*100/len(acc)

        def get_op_accuracy(gt_pred_op):
            accuracy = []
            for gt, pred in gt_pred_op:
                acc = 1
                for i in range(len(gt)):
                    if pred[i] != gt[i]: acc = 0
                accuracy.append(acc)
            return sum(accuracy)*100/len(accuracy)

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## get accuracy of predicting aggregate function
        gt_agg = list(self.df['gt_agg'].values)
        pred_agg = list(self.df['pred_agg'].values)
        gt_pred_agg = zip(gt_agg, pred_agg)
        agg_acc = get_agg_func_accuracy(gt_pred_agg)

        ## get accuracy of predicting operator
        gt_op = list(self.df['gt_op'].values)
        pred_op = list(self.df['pred_op'].values)
        gt_pred_op= zip(gt_op, pred_op)
        op_acc = get_op_accuracy(gt_pred_op)

        return round(agg_acc, 2), round(op_acc, 2)

