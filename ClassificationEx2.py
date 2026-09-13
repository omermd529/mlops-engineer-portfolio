from sklearn.datasets  import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score

X,y = make_classification(
    n_samples=5000,
    n_features=10,
    weights=[0.97,0.03],
    random_state=42
    
)

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

fraud_proba = model.predict_proba(X_test)[:,1]

# Test different thresholds - same model, very different behaviour

print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Blocked':>8}")

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for threshold in thresholds:
    predicted=(fraud_proba >= threshold).astype(int)
    
    
    # Avoid division by zero when no positives predicted
    
    if predicted.sum() == 0:
        print(f"{threshold:>10.1f} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {predicted.sum():>8}")
        continue
    precision = precision_score(y_test, predicted, zero_division=0)
    recall= recall_score(y_test, predicted, zero_division=0)
    f1 = f1_score(y_test, predicted, zero_division=0)
    blocked = predicted.sum()
    
    print(f"{threshold:>10.1f} | {precision:>10.4f} | {recall:>8.4f} | {f1:>8.4f} | {blocked:>8}")
    
    