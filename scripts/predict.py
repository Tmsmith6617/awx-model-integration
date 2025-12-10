import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# For scikit-learn specifically
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pickle
import pandas as pd
import sys

# Accept CSV input from playbook
input_file = sys.argv[1]  # e.g., "files/input.csv"

# Load data
data = pd.read_csv(input_file)

# Load your model
with open("models/my_model.pk1", "rb") as f:
    model = pickle.load(f)

# Run predictions
predictions = model.predict(data)

# Print predictions to stdout so AWX can capture
print(predictions.tolist())
