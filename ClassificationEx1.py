from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Generate synthetic binary classification data
# Think of this as : 1000  transactions, 10 features each

X,y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.97,0.03], # 97% Not fraud, 3% Fraud - realistic imbalance
    random_state=42
)

X_train,X_test,y_train,y_test= train_test_split(
    X,y,test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

# OUTPUT 1: Hard predictions 0 or 1

predictions=model.predict(X_test)
print("Hard Predictions (first 20):")
print(predictions[:20])

#OUTPUT 2: Probabilities what you actually use in production
probabilities=model.predict_proba(X_test)
print("\nProbability output shape:" , probabilities.shape)
print("First 5 rows - [prob_not_fraud, prob_fraud]:")
print(probabilities[:5])

# OUTPUT 3: just the fraud probability column
fraud_proba = probabilities[:, 1]
print("\nFraud probability only(first 10):")
print(fraud_proba[:10])

# Build a results DataFrame - This is what your serving API returns.

results = pd.DataFrame({
    'actual': y_test,
    'predicted': predictions,
    'fraud_proba': fraud_proba,
    'decision': ['BLOCK'if p > 0.5 else 'ALLOW'for p in fraud_proba]
    
    
})

print("\nResults sample:")
print(results.head(10))
