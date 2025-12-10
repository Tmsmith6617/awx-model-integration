import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# For scikit-learn specifically
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pickle
import pandas as pd
import sys

# Accept CSV input and output from AWX playbook
input_file = sys.argv[1]  # e.g., "files/input.csv"
output_file = sys.argv[2]  # e.g., "files/output.csv"

# Load data
data = pd.read_csv(input_file)

# Load your trained model
with open("models/my_model.pk1", "rb") as f:
    model = pickle.load(f)

# Ensure input has the exact columns the model was trained on (excluding target)
feature_columns = ['Time', 'Length', 'Protocol_BROWSER', 'Protocol_ICMP',
                   'Protocol_NBNS', 'Protocol_TCP', 'Protocol_TLSv1.2']

X = data[feature_columns]

# Run predictions
predictions = model.predict(X)

# Save predictions to output CSV
output_df = data.copy()
output_df['prediction'] = predictions
output_df.to_csv(output_file, index=False)
