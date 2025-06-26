# 📊 My End-to-End Machine Learning Workflow

I use a structured, reproducible pipeline to go from raw data to deployed ML models.

---

## 1. Data Ingestion and Merging

- Read data from CSVs, APIs, or databases  
- Merge and stack multiple sources

```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.merge(df1, df2, on='id')
df = pd.concat([df1, df2], axis=0)
```

---

## 2. 🔍 Data Exploration & Profiling

I read the data dictionary and try to understand the data before making changes:

```python
df.info()
df.describe().T
df.describe(include='object').T
df.isnull().sum()
df.duplicated().sum()
```

Visualize relationships and distributions:

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
```

---

## 3. 🧼 Data Cleaning

Standardize and sanitize data:

- Rename columns for consistency  
- Remove duplicates  
- Format strings, dates, etc.

```python
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df.drop_duplicates(inplace=True)
df.isin([0]).sum()  # Check for invalid values
```

---

## 4. ⚠️ Outlier Detection & Handling

Handle extreme values before scaling/imputation:

**IQR Method**

```python
import numpy as np

def outlier_bounds(col):
    Q1, Q3 = np.percentile(col.dropna(), [25, 75])
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
```

**Z-Score Method**

```python
from scipy.stats import zscore

df['z'] = zscore(df['income'])
df.loc[df['z'].abs() > 3, 'income'] = df['income'].median()
```

---

## 5. 🛠️ Feature Engineering

Create new, useful features from raw data:

```python
df['log_income'] = np.log1p(df['income'])
df['income_per_age'] = df['income'] / (df['age'] + 1)

df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60], labels=['20s', '30s', '40s', '50s'])
```

---

## 6. ✂️ Train/Test Split

Important: Split before scaling/encoding to prevent data leakage.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 7. ⚙️ Preprocessing: Scaling & Encoding

Apply only on training data, then transform test data.

**Standard Scaling**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**One-Hot Encoding**

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['color']])
```

---

## 8. 🔍 Feature Selection

Reduce dimensionality and overfitting risk.

**Variance Threshold**

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_train_reduced = selector.fit_transform(X_train_scaled)
```

**Mutual Information**

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train)
```

---

## 9. 🤖 Modeling & Evaluation

Train the model and assess performance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

**Cross-validation**

```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
```

**Explainability**

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test_scaled, y_test)
```

---

## 10. 🔧 Hyperparameter Tuning

Improve performance with grid or randomized search:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(LogisticRegression(), param_grid={'C': [0.1, 1, 10]}, cv=5)
grid.fit(X_train_scaled, y_train)
```

---

## 11. 🚀 Model Export & Deployment

**Export Model**

```python
import joblib

joblib.dump(model, 'model.pkl')
```

**Deploy with FastAPI / Streamlit / Flask**  
Example with joblib + FastAPI (deployment logic omitted for brevity).

---

## 12. 🔁 Pipeline Automation & Versioning

**Build Reusable Pipelines**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

**Track Versions**

Use MLflow, DVC, or Weights & Biases to version experiments, models, and datasets.  
Log metadata, inputs, and outputs consistently.

---

## ✨ My guiding Principles

- 🔐 **Reproducibility**: Always set `random_state`  
- ⚠️ **No Data Leakage**: Split before transform  
- 🧱 **Modular Design**: Reuse components via pipelines  
- 📊 **Visual Intuition**: Use plots early, metrics later  
- 💾 **Version Everything**: Data, code, models
