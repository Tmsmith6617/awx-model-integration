import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pickle
import pandas as pd
import sys

input_file = sys.argv[1]

# Load data
data = pd.read_csv(input_file)

# Drop target column if exists
if 'bad_packet' in data.columns:
    data = data.drop(columns=['bad_packet'])

# Ensure columns are in the correct order for the model
expected_cols = ['Time', 'Length', 'Protocol_BROWSER', 'Protocol_ICMP', 
                 'Protocol_NBNS', 'Protocol_TCP', 'Protocol_TLSv1.2']
data = data[expected_cols]

# Load model
with open("models/my_model.pk1", "rb") as f:
    model = pickle.load(f)

# Run predictions
predictions = model.predict(data)

# Print predictions
print(predictions.tolist())
