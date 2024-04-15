import pickle
import pandas as pd
from scipy.stats import pearsonr

# Load WSI representations
def load_wsi_representations(filename):
    with open(filename, 'rb') as f:
        wsi_representations = pickle.load(f)
    return wsi_representations

wsi_representations = load_wsi_representations("wsi_representation.pkl")

# Load transcriptomic data
def load_transcriptomic_data(filename):
    transcriptomic_data = pd.read_csv(filename)
    return transcriptomic_data

transcriptomic_data = load_transcriptomic_data("transcriptomic_data.csv")

# Run Pearson correlation between WSI representations and transcriptomic data
correlations = {}
for slide_name, representation in wsi_representations.items():
    for column in transcriptomic_data.columns:
        corr, p_value = pearsonr(representation, transcriptomic_data[column])
        if abs(corr) > 0.7 and p_value < 0.01:
            if slide_name not in correlations:
                correlations[slide_name] = []
            correlations[slide_name].append((column, corr, p_value))

# Save list of correlated features
correlated_features = {}
for slide_name, data in correlations.items():
    correlated_features[slide_name] = [feature[0] for feature in data]

with open("outputs/correlated_features.pkl", 'wb') as f:
    pickle.dump(correlated_features, f)