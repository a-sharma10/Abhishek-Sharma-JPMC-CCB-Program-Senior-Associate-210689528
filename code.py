import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 1. DATA LOADING & INITIAL CLEANING
# Load column names from the external metadata file
with open('census-bureau.columns', 'r') as f:
    columns = [line.strip() for line in f.readlines()]

# na_values=['?'] ensures we treat true unknowns as NaN for imputation.
# 'Not in universe' is preserved as a valid string category representing 'Not Applicable'.
df = pd.read_csv('census-bureau.data', 
                 header=None, 
                 names=columns, 
                 skipinitialspace=True, 
                 na_values=['?'])

# Encode target: Map '50000+.' to 1 (High Income) and others to 0 (Low Income)
df['label_numeric'] = df['label'].apply(lambda x: 1 if '+' in str(x) else 0)

# 2. FEATURE SELECTION
# Numerical features chosen for high correlation with earning potential
num_features = [
    'age', 'wage per hour', 'capital gains', 'capital losses', 
    'dividends from stocks', 'num persons worked for employer', 'weeks worked in year'
]

# Categorical features chosen to define life-stage and employment stability
cat_features = [
    'education', 'major occupation code', 'major industry code', 'marital stat', 
    'sex', 'race', 'full or part time employment stat', 
    'detailed household summary in household'
]

# 3. PREPROCESSING PIPELINE
# Handle numerical data: Impute missing with median (outlier robust) and scale to unit variance
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Handle categorical data: Impute true unknowns with mode and One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# 4. TASK 1: CLASSIFICATION (INCOME PREDICTION)
# scale_pos_weight=3 balances the heavy majority of <50k earners
clf_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=3, 
    eval_metric='logloss',
    random_state=42
)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf_model)
])

# Split data with stratification to maintain class ratios in train/test sets
X = df[num_features + cat_features]
y = df['label_numeric']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("--- Task 1: Classification Performance ---")
print(classification_report(y_test, y_pred))

# 5. TASK 2: SEGMENTATION (CUSTOMER CLUSTERING)
# Process features for distance-based clustering
X_seg_processed = preprocessor.fit_transform(df[num_features + cat_features])

# Segment into 3 distinct marketing personas
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_seg_processed)

# 6. MARKETING ANALYTICS & PROFILING
print("\n--- Task 2: Marketing Segment Profiles ---")
profile = df.groupby('segment')[['age', 'capital gains', 'weeks worked in year']].mean()
print(profile)

for i in range(3):
    top_edu = df[df['segment'] == i]['education'].mode()[0]
    print(f"Segment {i} Primary Education Level: {top_edu}")
