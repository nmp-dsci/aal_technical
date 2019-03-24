
"""
    INITIALISE mapping of input table

df_dtypes = df.dtypes.astype(str)

for col in df.columns:
    dtype = {'dtype':df_dtypes[col]}
    nuniq = {'nuniq':df[col].nunique()}
    missing_count = {'missing':df[col].apply(lambda x: 1 if x != x or x=="?" else 0).sum()}
    levels = {'levels':[]}
    if nuniq.get('nuniq') <= 25: 
        levels = {'levels':list(df[col].unique())}
    # 
    print({**{'column':col},**dtype,**nuniq,**levels,**missing_count})


"""

import numpy as np
import pandas as pd


class column_utils:
    #
    levels_move = ['No', 'Steady', 'Down', 'Up']
    #
    def treatment_ordinal(series=pd.Series(),levels=[]):
        level_mapping = dict(zip(levels,[x for x in range(len(levels))]))
        new_series = series.replace(level_mapping).astype(int)
        return new_series
    def treatment_categorical(col='aaa', df=pd.DataFrame(),levels=[]):
        return pd.get_dummies(df[col],prefix=col,prefix_sep='__').drop(col+'__?',axis=1,errors='ignore').fillna(0)
    def treatment_NLP(col='diag_2_desc',df=pd.DataFrame(),topn= 100):
        nlp_raw = df[col].str.lower().str.strip()
        nlp_raw = nlp_raw.str.split('[^a-z0-9]',expand=True).reset_index().melt(id_vars = 'index')
        keep_values = nlp_raw.value.value_counts().head(topn).reset_index().drop('value',axis=1).rename(columns={'index':'value'})
        nlp_raw = nlp_raw.merge(keep_values,on='value',how='inner').drop('variable',axis=1)
        nlp_raw['dummy'] = 1
        nlp_raw = nlp_raw.groupby(['index','value'])['dummy'].max().unstack('value').fillna(0)
        nlp_raw.columns = [col+'__'+x for x in nlp_raw.columns]
        return  nlp_raw




table_definition = pd.DataFrame([
        {'column': 'rowID', 'dtype': 'int64', 'nuniq': 10000, 'levels': [],'format':'numeric'}
    ,    {'column': 'race', 'dtype': 'object', 'nuniq': 6
            ,   'levels': ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Other', 'Asian']
            ,   'format':'categorical','needs_work':True,'missing_var':'?', 'missing': 221}
    ,    {'column': 'gender', 'dtype': 'object', 'nuniq': 2, 'levels': ['Female', 'Male'],'format':'dummy'}
    ,    {'column': 'age', 'dtype': 'object', 'nuniq': 10
            ,   'levels': ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
            ,   'format':'ordinal'}
    ,    {'column': 'weight', 'dtype': 'object', 'nuniq': 8
            ,   'levels': ['[0-25)','[25-50)','[50-75)','[75-100)','[100-125)','[125-150)','[150-175)']
            ,   'format':'drop','needs_work':True,'missing_var':'?', 'missing': 9592}
    ,    {'column': 'admission_type_id', 'dtype': 'object', 'nuniq': 6
            ,   'levels': ['Elective', 'Urgent', 'Not Available', 'Emergency', 'Not Mapped', 'Newborn']
            ,   'format':'categorical','needs_work':True,'missing_var':np.NaN, 'missing': 721}
    ,    {'column': 'discharge_disposition_id', 'dtype': 'object', 'nuniq': 21
            ,   'levels': ['Discharged to home', 'Discharged/transferred to home with home health service', 'Expired'
                ,   'Discharged/transferred to a long term care hospital.', 'Discharged/transferred to SNF'
                ,   'Discharged/transferred to another  type of inpatient care institution', 'Not Mapped'
                ,   'Discharged/transferred to another short term hospital', np.NaN, 'Left AMA'
                ,   'Discharged/transferred to another rehab fac including rehab units of a hospital.'
                ,   'Hospice / medical facility', 'Hospice / home', 'Discharged/transferred/referred to a psychiatric hospital of a psychiatric distinct part unit of a hospital'
                ,   'Discharged/transferred to ICF', 'Discharged/transferred to home under care of Home IV provider'
                ,   'Admitted as an inpatient to this hospital', 'Discharged/transferred/referred another institution for outpatient services'
                ,   'Discharged/transferred to a federal health care facility.', 'Discharged/transferred within this institution to Medicare approved swing bed'
                ,   'Discharged/transferred/referred to this institution for outpatient services'
                ,   'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare']
            ,   'format':'NLP','needs_work':True,'missing_var':np.NaN, 'missing': 469}
    ,    {'column': 'admission_source_id', 'dtype': 'object', 'nuniq': 10
            ,   'levels': ['Physician Referral', 'Transfer from another health care facility', 'Emergency Room'
                ,   'Transfer from a Skilled Nursing Facility (SNF)', 'Transfer from a hospital', 'Not Mapped', 'Clinic Referral'
                ,   'HMO Referral', 'Not Available', 'Court/Law Enforcement']
            ,   'format':'categorical','needs_work':True,'missing_var':np.NaN, 'missing': 936}
    ,    {'column': 'time_in_hospital', 'dtype': 'int64', 'nuniq': 14, 'levels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],'format':'numeric'}
    ,    {'column': 'payer_code', 'dtype': 'object', 'nuniq': 16
            ,   'levels': ['CP', 'UN', 'MC', 'HM', 'SP', 'CM', 'BC', 'MD', 'WC', 'OG', 'PO', 'DM', 'SI', 'OT', 'CH']
            ,   'format':'categorical','needs_work':True,'missing_var':'?', 'missing': 5341}
    ,    {'column': 'medical_specialty', 'dtype': 'object', 'nuniq': 53, 'levels': []
            ,   'format':'NLP','needs_work':True,'missing_var':np.NaN, 'missing': 4100}
    ,    {'column': 'num_lab_procedures', 'dtype': 'int64', 'nuniq': 108, 'levels': [],'format':'numeric'}
    ,    {'column': 'num_procedures', 'dtype': 'int64', 'nuniq': 7, 'levels': [ 0, 1, 2, 3,4, 5, 6],'format':'numeric'}
    ,    {'column': 'num_medications', 'dtype': 'int64', 'nuniq': 68, 'levels': [],'format':'numeric'}
    ,    {'column': 'number_outpatient', 'dtype': 'int64', 'nuniq': 23
            ,   'levels': [0, 2, 1, 4, 3, 7, 5, 6, 19, 13, 8, 18, 9, 10, 16, 17, 12, 36, 14, 27, 11, 21, 15],'format':'numeric'}
    ,    {'column': 'number_emergency', 'dtype': 'int64', 'nuniq': 12, 'levels': [0, 1, 2, 3, 4, 6, 42, 5, 7, 8, 13, 9],'format':'numeric'}
    ,    {'column': 'number_inpatient', 'dtype': 'int64', 'nuniq': 11, 'levels': [0, 1, 2, 4, 3, 7, 5, 6, 9, 8, 10],'format':'numeric'}
    ,    {'column': 'diag_1', 'dtype': 'object', 'nuniq': 458, 'levels': [],'format':'drop','needs_work':True,'missing_var':'?', 'missing': 2}
    ,    {'column': 'diag_2', 'dtype': 'object', 'nuniq': 430, 'levels': [],'format':'drop','needs_work':True,'missing_var':'?', 'missing': 59}
    ,    {'column': 'diag_3', 'dtype': 'object', 'nuniq': 461, 'levels': [],'format':'drop','needs_work':True,'missing_var':'?', 'missing': 208}
    ,    {'column': 'number_diagnoses', 'dtype': 'int64', 'nuniq': 9, 'levels': [9, 6, 3, 7, 8, 5, 4, 2, 1],'format':'numeric'}
    ,    {'column': 'max_glu_serum', 'dtype': 'object', 'nuniq': 4, 'levels': ['None', 'Norm', '>200', '>300'],'format':'ordinal'}
    ,    {'column': 'A1Cresult', 'dtype': 'object', 'nuniq': 4, 'levels': ['None', 'Norm', '>7', '>8'],'format':'ordinal'}
    ,    {'column': 'metformin', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'repaglinide', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'nateglinide', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'chlorpropamide', 'dtype': 'object', 'nuniq': 3, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glimepiride', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'acetohexamide', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glipizide', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glyburide', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'tolbutamide', 'dtype': 'object', 'nuniq': 2, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'pioglitazone', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'rosiglitazone', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'acarbose', 'dtype': 'object', 'nuniq': 3, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'miglitol', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'troglitazone', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'tolazamide', 'dtype': 'object', 'nuniq': 2, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'examide', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'citoglipton', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'insulin', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glyburide.metformin', 'dtype': 'object', 'nuniq': 4, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glipizide.metformin', 'dtype': 'object', 'nuniq': 2, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'glimepiride.pioglitazone', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'metformin.rosiglitazone', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'metformin.pioglitazone', 'dtype': 'object', 'nuniq': 1, 'levels': column_utils.levels_move,'format':'categorical'}
    ,    {'column': 'change', 'dtype': 'object', 'nuniq': 2, 'levels': ['No', 'Ch'],'format':'dummy','yes_var':'Ch'}
    ,    {'column': 'diabetesMed', 'dtype': 'object', 'nuniq': 2, 'levels': ['No', 'Yes'],'format':'dummy','yes_var':'Yes'}
    ,    {'column': 'readmitted', 'dtype': 'bool', 'nuniq': 2, 'levels': [False, True],'format':'dummy','yes_var':True}
    ,    {'column': 'diag_1_desc', 'dtype': 'object', 'nuniq': 457, 'levels': [],'format':'NLP','needs_work':True,'missing_var':np.NaN, 'missing': 2}
    ,    {'column': 'diag_2_desc', 'dtype': 'object', 'nuniq': 429, 'levels': [],'format':'NLP','needs_work':True,'missing_var':np.NaN, 'missing': 59}
    ,    {'column': 'diag_3_desc', 'dtype': 'object', 'nuniq': 460, 'levels': [],'format':'NLP','needs_work':True,'missing_var':np.NaN, 'missing': 208}
])


