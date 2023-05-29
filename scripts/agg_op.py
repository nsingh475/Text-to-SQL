import sys
mode = sys.argv[1] ## getting argument from commandline

from library.aggregate_operator import *

out_path = f'Data/Output/{mode}/'
rule_path = 'Data/rules/'
agg_file = 'aggregate_rules'
op_file = 'operator_rules'
wiki_file = 'WikiSQL_processed.csv'

## Predicting Aggregate and Operator
pred_agg_op_Obj = Predict_AggregateFunction_and_Operator(rule_path, out_path, agg_file, op_file, wiki_file)
df = pred_agg_op_Obj.run() 

## Evaluating 
eval_agg_op_Obj = Evaluate_AggregateFunction_and_Operator(df)
agg_acc, op_acc = eval_agg_op_Obj.run() 

print('---------------------- Aggregate Function and Operator Prediction Evaluation Report ----------------------')
print(f'Aggregate Function prediction accuracy: {agg_acc}')
print(f'Operator prediction Accuracy: {op_acc}')