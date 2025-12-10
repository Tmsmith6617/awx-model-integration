import pickle
import pandas as pd
import sys

# Accept a CSV file as input from the playbook
input_file = sys.argv[1]  # e.g., "/tmp/input.csv"
output_file = sys.argv[2]  # e.g., "/tmp/output.csv"

# Load data
data = pd.read_csv(input_file)

# Load your model
with open("models/my_model.pk1", "rb") as f:
    model = pickle.load(f)

# Run predictions
predictions = model.predict(data)

# Save results
data["prediction"] = predictions
data.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
