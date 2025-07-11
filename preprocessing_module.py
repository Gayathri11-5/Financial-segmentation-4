from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
def preprocess_data(data):
    data = data.copy()

    # Convert any object-looking numbers to numeric
    data = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='ignore'))

    # Drop columns
    for col in data.columns:
        if 'Invoice ID' in col:
            data.drop(col, axis=1, inplace=True)
    if 'Date' in data.columns:
        data.drop('Date', axis=1, inplace=True)
    if 'Time' in data.columns:
        data.drop('Time', axis=1, inplace=True)

    # Create feature if possible
    v1 = ['Quantity', 'gross income', 'Unit price']
    if all(col in data.columns for col in v1):
        data["spending_score"] = data[v1].sum(axis=1)

    # Numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found after conversion.")

    # Imputation
    imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # Encode categoricals
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Re-select numeric columns after encoding
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data[numeric_cols])

    return data, scaled_array
