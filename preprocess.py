import pandas as pd

def clean_data(df):
    # Replace 'N/A' with actual NaN
    df = df.replace("N/A", pd.NA)

    # Drop missing values
    df = df.dropna()

    # Drop ID column (not useful for ML)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Convert categorical → numeric
    df = pd.get_dummies(df)

    return df