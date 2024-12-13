import pandas as pd

# Load the dataset
data_path = r'C:\Users\anura\Desktop\Tobacco Morality Prediction and Analysis\Notebooks\integrated_tobacco_data.csv'
df = pd.read_csv(data_path)

# 1. Calculate Tobacco Affordability
df["Tobacco Affordability"] = df["Value_x"] / df["Value_y"]

# 2. Calculate Smoking Cessation Success Rates
# Combine prescription costs and normalize by the population aged 16 and over
df["Cessation Success Rate"] = (df["Net Ingredient Cost of Bupropion (Zyban)"] + 
                                 df["Net Ingredient Cost of Varenicline (Champix)"]) / df["16 and Over"]

# 3. Smoking Prevalence Trends
# Sum up smoking prevalence across all age groups
age_columns = ["16-24", "25-34", "35-49", "50-59", "60 and Over"]
df["Smoking Prevalence"] = df[age_columns].sum(axis=1)

# 4. Normalize Financial Data for Inflation
# Using the first year's values (2004) as the baseline for normalization
base_year = df.loc[df["Year"] == 2004]

# Apply normalization
df["Normalized Income (Value_x)"] = df["Value_x"] / base_year["Value_x"].values[0]
df["Normalized Cost (Value_y)"] = df["Value_y"] / base_year["Value_y"].values[0]
df["Normalized Bupropion Cost"] = df["Net Ingredient Cost of Bupropion (Zyban)"] / base_year["Net Ingredient Cost of Bupropion (Zyban)"].values[0]
df["Normalized Varenicline Cost"] = df["Net Ingredient Cost of Varenicline (Champix)"] / base_year["Net Ingredient Cost of Varenicline (Champix)"].values[0]

# Save the engineered dataset
# Correct the output path to include the filenme
output_path = r"C:\Users\anura\Desktop\Tobacco Morality Prediction and Analysis\Notebooks\output.csv"


# Save the engineered dataset
df.to_csv(output_path, index=False)

print(f"Feature engineering completed. Saved at {output_path}")
