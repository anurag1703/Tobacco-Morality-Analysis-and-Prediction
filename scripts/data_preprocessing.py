import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load your dataset
data_path = r""
df = pd.read_csv(data_path)

# Define features and target
target = "Value_y"  # Adjust target column
features = df.drop(columns=["Year", target])  # Drop non-informative columns

# Step 1: Check and Replace Infinite Values
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Step 2: Handle NaN Values
imputer = SimpleImputer(strategy="median")
features_imputed = imputer.fit_transform(features)

# Step 3: Scale Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Define X and y
X = features_scaled
y = df[target]

print("Data preprocessing complete.")
