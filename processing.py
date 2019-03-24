"""
Key questions to answer with the dataset
a) What are some important indicators of whether a patient will be readmitted? 
b) What could the hospital system do with this information? 

Steps taken in the code below
    1. data processing: clean dataset 



"""


import pandas as pd
import numpy as np
import re,time,os
import time,datetime,math
import json

pd.options.display.max_columns = 100

from config import * 

df = pd.read_csv('10kDiabetes.zip',compression='zip')

####################################################
## STEP 1: Data Processing

# Prepard dataset for modelling
table_definition.format.value_counts()

clean_df = pd.DataFrame([],index=df.index.values)

# categorical treatment
vars_categorical = json.loads(table_definition.query('format=="categorical"').T.to_json())
for idx in vars_categorical:
    col_attr = pd.Series(vars_categorical.get(idx))
    print('Categorical varible: "{b}"'.format(b=col_attr.column))
    clean_df = pd.concat([clean_df
        ,   column_utils.treatment_categorical(col_attr.column,df,col_attr.levels)
    ],axis=1,sort=False)

assert clean_df.isnull().sum().sum() == 0, "ERROR for Categorical treatment, letting through NULLs"

# numeric treatment
vars_numeric = json.loads(table_definition.query('format=="numeric"').T.to_json())
for idx in vars_numeric:
    col_attr = pd.Series(vars_numeric.get(idx))
    print('Numeric varible: "{b}"'.format(b=col_attr.column))
    clean_df = pd.concat([clean_df,df[[col_attr.column]]],axis=1,sort=False)    

assert clean_df.isnull().sum().sum() == 0, "ERROR for Numeric treatment, letting through NULLs"


# ordinal treatment
vars_ordinal = json.loads(table_definition.query('format=="ordinal"').T.to_json())
for idx in vars_ordinal:
    col_attr = pd.Series(vars_ordinal.get(idx))
    print('Ordinal varible: "{b}"'.format(b=col_attr.column))
    clean_df = pd.concat([clean_df,pd.DataFrame({col_attr.column:column_utils.treatment_ordinal(df[col_attr.column],col_attr.levels)})],axis=1,sort=False)  

assert clean_df.isnull().sum().sum() == 0, "ERROR for Ordinal treatment, letting through NULLs"

# NLP treatment
vars_NLP = json.loads(table_definition.query('format=="NLP"').T.to_json())
for idx in vars_NLP:
    col_attr = pd.Series(vars_NLP.get(idx))
    print('NLP varible: "{b}"'.format(b=col_attr.column))
    clean_df = pd.concat([clean_df,column_utils.treatment_NLP(col_attr.column,df,topn=50)],axis=1,sort=False)

clean_df = clean_df.fillna(0)

assert clean_df.isnull().sum().sum() == 0, "ERROR for NLP treatment, letting through NULLs"

# Dummy treatment
vars_dummy = json.loads(table_definition.query('format=="dummy"').T.to_json())
for idx in vars_dummy:
    col_attr = pd.Series(vars_dummy.get(idx))
    print('Dummy varible: "{b}"'.format(b=col_attr.column))
    clean_df = pd.concat([clean_df,pd.DataFrame({col_attr.column:df[col_attr.column].apply(lambda x: 1 if x==col_attr.yes_var else 0)})],axis=1,sort=False)  

assert clean_df.isnull().sum().sum() == 0, "ERROR for Dummy treatment, letting through NULLs"

####################################################
# STEP 2: modelling for most important factors to 'readmitted' patients
## Build column format for 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier



## column definition
column_df = pd.DataFrame({'column':list(clean_df.columns.values)})

response = 'readmitted'
column_df['predictor'] = column_df.column.apply(lambda x: 1 if x !=response else 0)

column_df['parent_column'] =column_df.column.str.split('__',expand=False).apply(lambda x: x[0] if len(x)==2 else np.NaN)

## test/train
clean_df['runif'] = np.random.uniform(size=clean_df.shape[0])
clean_df['train'] = clean_df.runif.apply(lambda x: 1 if x < 0.8 else 0)


## train model for variable importance

rf = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state=0,min_samples_split=50,verbose=1)
gbm = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None,
    max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
    validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

## train model
importance_df = pd.DataFrame({},index=column_df.query('predictor==1').column.values)

for classifier in ['rf','gbm']:
    print('Fitting classifer: "{}"'.format(classifier))
    globals().get(classifier).fit(clean_df.query('train==1')[list(column_df.query('predictor==1').column.values)], clean_df.query('train==1')[response])
    ## review fit
    pred_name = '{}_pred'.format(classifier)
    clean_df[pred_name] = globals().get(classifier).predict(clean_df[list(column_df.query('predictor==1').column.values)])
    ## get diagnositics
    diag_df = clean_df.groupby(['train',response,pred_name]).size().reset_index().rename(columns={0:'n'})
    diag_df = diag_df.merge(clean_df.groupby(['train',pred_name]).size().reset_index().rename(columns={0:'all'}),on=['train',pred_name],how='left')
    diag_df['accuracy'] = 1.0 * diag_df['n']/diag_df['all']
    print(diag_df.groupby(['train',response,pred_name])['accuracy'].max().unstack(pred_name))
    globals()["{}_diag_df".format(classifier)] = diag_df
    ## Importace
    importance_df[classifier+'_imp'] = pd.Series(globals().get(classifier).feature_importances_,index=column_df.query('predictor==1').column.values)
    importance_df[classifier+'_rank'] = importance_df[classifier+'_imp'].rank(ascending=False,method='first')



## importance
importance_df.sort_values('rf_rank',ascending=True).head(20)
importance_df.sort_values('gbm_rank',ascending=True).head(20)


####################################################
## STEP 3: review top factors
df[response].value_counts(normalize=True)


#number_inpatient
var = 'number_inpatient'
pd.concat([pd.crosstab(df[var], df[response],normalize='index'),df[var].value_counts()],axis=1)

#number_diagnoses
var = 'number_diagnoses'
pd.concat([pd.crosstab(df[var], df[response],normalize='index'),df[var].value_counts()],axis=1)

#number_outpatient
var = 'number_outpatient'
cuts = 7
pd.concat([pd.crosstab(pd.cut(df[var],cuts), df[response],normalize='index'),pd.cut(df[var],cuts).value_counts()],axis=1)

#num_medications [numeric variable]
var = 'num_medications'

pd.concat([pd.crosstab(pd.cut(df[var],10), df[response],normalize='index'),pd.cut(df[var],10).value_counts()],axis=1)


#number_outpatient
var = 'discharge_disposition_id'
levels = ['expired','to','discharged',]
pd.concat([pd.crosstab(df[var], df[response],normalize='index'),df[var].value_counts()],axis=1)




