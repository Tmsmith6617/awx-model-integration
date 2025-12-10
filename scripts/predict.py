import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pickle
import pandas as pd
import sys

# Accept a CSV file as input from the playbook
input_file = sys.argv[1]
output_file = sys.argv[2]

# Load data
data = pd.read_csv(input_file)
print(f"Read {len(data)} rows from {input_file}")

# Load your model
with open("models/my_model.pk1", "rb") as f:
    model = pickle.load(f)

# Run predictions
predictions = model.predict(data)

# Save predictions to a CSV
pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")
