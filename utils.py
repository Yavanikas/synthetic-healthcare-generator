import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def add_noise(data, noise_level=0.1):
    noisy = data.copy()

    for col in noisy.columns:
        if noisy[col].dtype != 'object':
            noisy[col] += np.random.normal(0, noise_level, len(noisy))

    return noisy

# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate(df, synthetic):
    target_col = df.columns[-1]

    # REAL DATA
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # SYNTHETIC DATA (SAFE)
    X_syn = synthetic.drop(columns=[target_col], errors='ignore')
    y_syn = synthetic.get(target_col, y)

    # convert synthetic target
    y_syn = (y_syn > 0.5).astype(int)

    # ALIGN COLUMNS (VERY IMPORTANT)
    X_syn = X_syn.reindex(columns=X.columns, fill_value=0)

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # REAL MODEL
    model_real = RandomForestClassifier()
    model_real.fit(X_train, y_train)
    real_score = accuracy_score(y_test, model_real.predict(X_test))

    # SYNTHETIC MODEL
    model_syn = RandomForestClassifier()
    model_syn.fit(X_syn, y_syn)
    syn_score = accuracy_score(y_test, model_syn.predict(X_test))

    return real_score, syn_score


# -------------------------------
# POSTPROCESS FUNCTION
# -------------------------------
def postprocess_synthetic(synthetic_df):
    df = synthetic_df.copy()

    for col in ["hypertension", "heart_disease", "stroke"]:
        if col in df.columns:
            df[col] = (df[col] > 0.5).astype(int)

    # NUMERICAL FIX
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].clip(lower=0)

    if "age" in df.columns:
        df["age"] = df["age"].clip(0, 100).round().astype(int)

    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].clip(10, 60).round(1)

    if "avg_glucose_level" in df.columns:
        df["avg_glucose_level"] = df["avg_glucose_level"].clip(50, 300).round(1)

    # REVERSE ENCODING
    if "gender_Female" in df.columns and "gender_Male" in df.columns:
        df["gender"] = df.apply(
            lambda row: "Female" if row["gender_Female"] > row["gender_Male"] else "Male",
            axis=1
        )
        df.drop(columns=["gender_Female", "gender_Male"], inplace=True)

    if "ever_married_Yes" in df.columns and "ever_married_No" in df.columns:
        df["ever_married"] = df.apply(
            lambda row: "Yes" if row["ever_married_Yes"] > row["ever_married_No"] else "No",
            axis=1
        )
        df.drop(columns=["ever_married_Yes", "ever_married_No"], inplace=True)

    work_cols = [c for c in df.columns if c.startswith("work_type_")]
    if work_cols:
        df["work_type"] = df[work_cols].idxmax(axis=1).str.replace("work_type_", "")
        df.drop(columns=work_cols, inplace=True)

    if "work_type" in df.columns:
        df["work_type"] = df["work_type"].replace({
            "children": "student/Unemployed"
        })    

    res_cols = [c for c in df.columns if c.startswith("Residence_type_")]
    if res_cols:
        df["Residence_type"] = df[res_cols].idxmax(axis=1).str.replace("Residence_type_", "")
        df.drop(columns=res_cols, inplace=True)

    smoke_cols = [c for c in df.columns if c.startswith("smoking_status_")]
    if smoke_cols:
        df["smoking_status"] = df[smoke_cols].idxmax(axis=1).str.replace("smoking_status_", "")
        df.drop(columns=smoke_cols, inplace=True)

    return df