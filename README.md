
# Titanic Dataset Cleaning & Preprocessing

This project demonstrates a complete pipeline to clean and preprocess the Titanic dataset, preparing it for machine learning or further analysis.

## Overview

The Titanic dataset contains information about passengers, including age, fare, gender, and whether they survived. This cleaning process handles missing data, encodes categorical variables, removes outliers, and scales numeric features.

## Steps

1. **Import Libraries:** pandas, numpy, matplotlib, seaborn, and sklearn's StandardScaler.  
2. **Load Data:** Read CSV with replacement for common missing value indicators.  
3. **Basic Data Inspection:** Check initial shape, head, missing values, and data types.  
4. **Handle Missing Values:**  
   - Fill missing `Age` with median age.  
   - Fill missing `Embarked` with the mode.  
   - Fill missing `Cabin` with "Unknown".  
5. **Encode Categorical Variables:**  
   - Map `Sex` to numeric values (`male` → 0, `female` → 1).  
   - One-hot encode `Embarked`.  
6. **Visualize Outliers:** Use boxplots for `Age` and `Fare`.  
7. **Remove Outliers:** Remove rows where `Age` or `Fare` fall outside 1.5*IQR.  
8. **Standardize Features:** Scale `Age` and `Fare` using StandardScaler.  
9. **Save Cleaned Data:** Export cleaned dataset as CSV.

## Usage

- Place your raw Titanic dataset CSV as `Titanic-Dataset.csv` in the same directory.  
- Run the script to produce a cleaned file named `Titanic-Dataset-Cleaned.csv`.  
- The cleaned dataset is ready for modeling or further analysis.

## Visualization

The script generates boxplots before and after outlier removal to help visually confirm the cleaning.

## Requirements

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 📁 Step 2: Load Data (replace blanks/NA with NaN)
df = pd.read_csv("Titanic-Dataset.csv", na_values=["", " ", "?", "NA", "n/a", "N/A"])

# 👀 Preview
print("Initial Shape:", df.shape)
print(df.head())

# 🔍 Step 3: Basic Data Info
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 🧼 Step 4: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

# ✅ Check again
print("\nMissing Values After Handling:\n", df.isnull().sum())

# 🔄 Step 5: Encode Categorical Columns
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 📊 Step 6: Visualize Outliers Using Boxplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='skyblue')
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color='lightgreen')
plt.title('Boxplot of Fare')

plt.tight_layout()
plt.show()

# ❌ Step 7: Remove Outliers Using IQR (for + if)
columns_to_check = ['Age', 'Fare']
df_cleaned = df.copy()

for col in columns_to_check:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    valid_indices = []
    for i in range(len(df_cleaned)):
        value = df_cleaned.iloc[i][col]
        if lower <= value <= upper:
            valid_indices.append(i)

    df_cleaned = df_cleaned.iloc[valid_indices].reset_index(drop=True)

print("\nShape After Removing Outliers:", df_cleaned.shape)

# 🎨 Optional: Compare boxplots before and after
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Fare'], color='salmon')
plt.title('Before Outlier Removal - Fare')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_cleaned['Fare'], color='lightgreen')
plt.title('After Outlier Removal - Fare')

plt.tight_layout()
plt.show()

# 📏 Step 8: Standardize 'Age' and 'Fare'
scaler = StandardScaler()
df_cleaned[['Age', 'Fare']] = scaler.fit_transform(df_cleaned[['Age', 'Fare']])

# 💾 Step 9: Save the Cleaned Dataset
df_cleaned.to_csv("Titanic-Dataset-Cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'Titanic-Dataset-Cleaned.csv'")

# 🧪 Final preview
print("\nCleaned Dataset Preview:")
print(df_cleaned.head())

*Happy Data Cleaning!*
