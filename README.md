# Formulating Text-to-SQL as a Question-Answering Task

Text-to-SQL interfaces allow users to interact with the databases without actually knowing any query  language. They provide a platform where at the frontend, a user can type-in their question in natural language (NLQ) and get the output on the screen and at the backend, the system converts the NLQ to SQL to generate the requested output. 

## Model Flowchart
![Flowchart](https://user-images.githubusercontent.com/87938938/205462318-d2e1895c-019a-440b-9e9d-161201429495.PNG)

## About the Project
This project formulates Text-to-SQL as a question answering task and adopts a slot-filling approach. I have worked on generating simple SQL query using WikiSQL dataset and the below SQL Template:

          SELECT <agg_fun>(<sel_col>)
          
          FROM tbl
          
          WHERE (<filter_col> <op> <value>)
          
Given a query in Natural Language (NLQ) and Table headers, Context is generated in the following format:

![image](https://user-images.githubusercontent.com/87938938/206928649-65bd8ca8-a5c4-4a10-bf97-8e9a116a2319.png)

For each slot in SQL Template, a question is generated in below format:

![image](https://user-images.githubusercontent.com/87938938/206928659-bff1ba65-9426-4847-8432-31b33a24c39e.png)

         
The answers for each question is predicted/extracted using three models (specialized in predicting a particular slot) summarized below:
- BERT based MRC: to predict select, filter column
- NER (using BIO labels): to extract multiple values for where clause from NLQ
- Text Classification (using ARM): to predict SQL aggregate function and operators


## Training Phase:
Training was done by running the following scripts in sequence:
- Pre-processing and encoding:  ```! python encoding.py 'train'```
- BERT Head Training: ```! python mrc.py 'train'```
- NER Training: ```! python ner.py 'train'```

## Testing Phase (Predicting and Evaluating):
Testing was done by running the following scripts in sequence:
- Pre-processing and encoding: ```! python encoding.py 'test'```
- BERT MRC prediction and evaluation: ```! python mrc.py 'test'```
- NER prediction and evaluation: ```! python ner.py 'test'```
- Aggregate and Operator prediction and evaluation: ```! python agg_op.py 'test'```



