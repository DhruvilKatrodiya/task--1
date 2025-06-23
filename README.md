
# Titanic Dataset Cleaning & Preprocessing

This project demonstrates a complete pipeline to clean and preprocess the Titanic dataset, preparing it for machine learning or further analysis.

---

## Overview

The Titanic dataset contains information about passengers, including age, fare, gender, and whether they survived. This cleaning process handles missing data, encodes categorical variables, removes outliers, and scales numeric features.

---

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

---

## Usage

- Place your raw Titanic dataset CSV as `Titanic-Dataset.csv` in the same directory.  
- Run the script to produce a cleaned file named `Titanic-Dataset-Cleaned.csv`.  
- The cleaned dataset is ready for modeling or further analysis.

---

## Visualization

The script generates boxplots before and after outlier removal to help visually confirm the cleaning.

---

## Requirements

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

Install missing packages via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Contact

For questions or suggestions, feel free to reach out.

---

*Happy Data Cleaning!*
