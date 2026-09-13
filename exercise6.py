import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib,json
from datetime import datetime

def check_feature_drift(reference_data, current_data,feature_names,threshold_pvalue=0.05):
    """Compares distribution of features between training data and live inference data. This is what Evidently AI does under the hood.
    KS test: if p-value < threshold, distribution has shifted significantly."""
    
    print("=" * 65)
    print("FEATURE DRIFT REPORT")
    print(f"Reference samples: {len(reference_data)} | Current samples: {len(current_data)}")
    print("=" * 65)
    
    drift_detected = False
    report = []
    
    print(f"\n{'Feature':<15} | {'Ref Mean':>10} | {'Cur Mean':>10} | {'KS Stat':>9} | {'P-Value':>9} | Status")
    print("-" * 75)
    
    for i,feature in enumerate(feature_names):
        ref_col = reference_data[:,i]
        cur_col = current_data[:,i]
        
        ks_stat, p_value = stats.ks_2samp(ref_col,cur_col)
        drifted = p_value < threshold_pvalue
        
        if drifted:
            drift_detected = True
            
        status = "DRIFT" if drifted else "OK"
        print(f"{feature:<15} | {ref_col.mean():>10.4f} | {cur_col.mean():>10.4f} | {ks_stat:>9.4f} | {p_value:>9.4f} | {status}")
        
        report.append({
            'feature':  feature,
            'ref_mean': float(ref_col.mean()),
            'cur_mean': float(cur_col.mean()),
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'drift_detected': bool(drifted)
        })
        
    print("\n" + "=" * 65)
    
    if drift_detected:
        print("ACTION: TRIGGER RETRAINING PIPELINE")
        print("Reason: Feature distribution has shifted from training baseline")
    else:
        print("STATUS: No significant drift detected. Model remains valid.")
        
    return drift_detected,report

def check_prediction_drift(reference_predictions,current_predictions, threshold=0.1):
    """Checks if the model's output distribution has shifted.
    If the model suddenly predicts much more or less fraud- something changed"""
    
    ref_fraud_rate = reference_predictions.mean()
    cur_fraud_rate = current_predictions.mean()
    delta = abs(cur_fraud_rate - ref_fraud_rate)
    
    print(f"\nPREDICTION DRIFT CHECK")
    print(f"Reference fraud rate: {ref_fraud_rate:.4f}")
    print(f"Current fraud rate: {cur_fraud_rate:.4f}")
    print(f"Delta: {delta:.4f} (threshold:{threshold})")
    
    
    if delta > threshold:
        print("STATUS: PREDICTION DRIFT DETECTED - investigate immediately")
        return True
    
    print("STATUS: Prediction distribution stable")
    
    return False

# Generate baseline training data 
X,y = make_classification(n_samples=5000,n_features=5, weights=[0.97,0.03], random_state=42)
feature_names = [f'feature_{i}'for i in range(5)]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
ref_predictions = model.predict_proba(X_test)[:,1]

# Simulate new inference data WITH drift -
# imagine transaction patterns changed after a holiday season

X_drifted, _ =make_classification(
    n_samples=500,
    n_features=5,
    weights=[0.95,0.05], # Fraud rate changed
    shift=1.5,
    random_state=99
)

print("\nScenario: 30 days of new inference data received\n")

feature_drift,report = check_feature_drift(X_test,X_drifted,feature_names)
cur_predictions = model.predict_proba(X_drifted)[:,1]
prediction_drift = check_prediction_drift(ref_predictions,cur_predictions)

if feature_drift or prediction_drift:
    print("\n>>> RETRAINING PIPELINE TRIGGERED <<<")
    print(">>> Next: Kubeflow Pipeline will run with new data <<<")
    
    #Save drift report - This goes to monitoring dashboard
    
    with open('drift_report.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'feature_drift': bool(feature_drift),
            'prediction_drift': bool(prediction_drift),
            'features': report
        },f,indent=2)
        
    print("Drift report saved to drift_report.json")
    