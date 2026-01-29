
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('loan.csv')

df.dropna(inplace=True)
categorical_features = ['gender', 'occupation', 'education_level', 'marital_status']
numerical_features = ['age', 'income', 'credit_score']
target = 'loan_status'

X = df[categorical_features + numerical_features]
y = df[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(model_pipeline, 'loan_model.joblib')
print("Model saved as loan_model.joblib")

joblib.dump(le, 'loan_status_label_encoder.joblib')
print("Label encoder for target variable saved as loan_status_label_encoder.joblib")
