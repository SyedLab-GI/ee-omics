import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cobra
import riptide
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Load expression matrix file
data = pd.read_csv('data/matrix.txt', sep='\t')

dfs = {
    'case': pd.DataFrame(),
    'control': pd.DataFrame()
}

# Get the 'gene_symbol' column
gene_symbol_column = data['gene_symbol']

# Filter columns for dfs_case based on 'EED' in the header
dfs['case'] = pd.concat([gene_symbol_column, data.filter(like='_eed_', axis=1)], axis=1)

# Filter columns for dfs_control based on 'control' in the header
dfs['control'] = pd.concat([gene_symbol_column, data.filter(like='_control_', axis=1)], axis=1)

# Create a map of patient ids
id_map = {
    'case': dfs['case'].columns[1:].tolist(),
    'control': dfs['control'].columns[1:].tolist()
}

# Gene Map with all the genes present in Recon3D
gene_name_num = pd.read_csv('data/gene_name_number.tsv', sep='\t')
gene_name_num = gene_name_num[['gene_number', 'symbol']]

for keys in dfs.keys():
    # Merge gene number annotations to dataframe
    dfs[keys] = pd.merge(gene_name_num, dfs[keys], left_on='symbol', right_on='gene_symbol')
    dfs[keys].drop('gene_symbol', axis=1, inplace=True)
    
# Load Cobra Model
model = cobra.io.load_matlab_model('data/Recon3D_301.mat')
cobra.util.array.create_stoichiometric_matrix(model).shape

# Run RIPTide for all patients
def run_riptide(fname, pat_name):
    transcript_abundances = riptide.read_transcription_file(fname,  norm=False)
    riptide_object = riptide.contextualize(model=model, transcriptome=transcript_abundances, fraction=1.)
    riptide_fva = riptide_object.flux_samples
    riptide_fva['sample_number'] = pat_name
    return riptide_fva

# Run for each patient
for keys in dfs.keys():
    for pat_id in id_map[keys]:
        dfs[keys][['gene_number', pat_id]].to_csv('sample.tsv', header=False, index=False, sep='\t')
        pat_output = run_riptide('sample.tsv', pat_id)
        pat_output.to_csv('outputs/{}.csv'.format(pat_id), index=False)

# Combine flux samples  
for keys in dfs.keys():
    list_of_pat_flux = []
    for pat_id in id_map[keys]:
        list_of_pat_flux.append(pd.read_csv('outputs/{}.csv'.format(pat_id)))
    pd.concat(list_of_pat_flux, join='inner', ignore_index=0).to_csv('outputs/riptide_{}_flux_sample.csv'.format(keys), index=False)
    
# Run a random forrest between case & control
df_case = pd.read_csv('outputs/riptide_case_flux_sample.csv')
df_control = pd.read_csv('outputs/riptide_control_flux_sample.csv')

df_case['label'] = 1
df_control['label'] = 0

df = pd.concat([df_case, df_control], join='inner')
df = df.reset_index(drop=True)

imp_list = []
accuracy_tracker = []
f1_tracker = []
precision_tracker = []
recall_tracker = []

val_distribution_tracker = []
imp_df = []

# Run model 100 times
for _ in range(100):
    # Keeping train-val split ratio 8:2
    test_sample = random.sample(list(df_case['sample_number'].unique()), 5) + random.sample(list(df_control['sample_number'].unique()), 5)

    train_sample = list(set(list(df_case['sample_number'].unique()) + list(df_control['sample_number'].unique())) - set(test_sample))
    
    X_train, y_train = df.loc[df['sample_number'].isin(train_sample), list(set(df.columns)-set(['sample_number', 'label']))].reset_index(drop=True), df.loc[df['sample_number'].isin(train_sample), ['label']].reset_index(drop=True)

    X_test, y_test = df.loc[df['sample_number'].isin(test_sample), list(set(df.columns)-set(['sample_number', 'label']))].reset_index(drop=True), df.loc[df['sample_number'].isin(test_sample), ['label']].reset_index(drop=True)

    y_train = y_train['label']
    y_test = y_test['label']
    
    # Train random forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)    
    selected_cols = X_train.columns[np.argsort(rf.feature_importances_)[::-1]]
    
    val_distribution_tracker.append(y_test.mean())
    f1_tracker.append(f1_score(y_test, y_test_pred))
    precision_tracker.append(precision_score(y_test, y_test_pred))    
    recall_tracker.append(recall_score(y_test, y_test_pred))        
    accuracy_tracker.append(accuracy_score(y_test, y_test_pred))
    gain = accuracy_score(y_test, y_test_pred)-max(y_test.mean(), (1-y_test.mean()))
    
    if gain>0.3:
        imp_list += list(selected_cols[:50])
        imp_df.append(pd.DataFrame({'reactions': selected_cols[:50], 'rank': range(1, 51)}))
        
# Top important reactions
average_reaction_rank = pd.DataFrame({'rank': pd.concat(imp_df).groupby('reactions')['rank'].mean()}).reset_index()
average_reaction_rank = average_reaction_rank.sort_values(by='rank').head(50).reset_index(drop=True)
average_reaction_rank.head(50).to_csv('outputs/average_reaction_rank.csv', index=False)

# Number of times differentiated
reaction_counts = pd.Series(imp_list).groupby(pd.Series(imp_list)).count().reset_index()
reaction_counts.columns = ['reactions', 'counts']
reaction_counts = reaction_counts.sort_values(by='counts', ascending=False)
reaction_counts.head(50).to_csv('outputs/reaction_counts.csv', index=False)