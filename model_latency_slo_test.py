import joblib
import numpy as np
import time
import sys
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_latency_slo_test(model_path,n_requests=1000,slo_p95_ms=50,slo_p99_ms=100):
    """Tests whether a model meets latency SLOs before deployment
    This runs in GitHub Actions before any KServe rollout.
    If the model is too slow,deployment is blocked automatically."""
    
    
    model= joblib.load(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Running {n_requests} inference requests...")
    print(f"SLO:p95 < {slo_p95_ms}ms,p99 < {slo_p99_ms}ms\n")
    
    #Generate test inputs
    X_test, _ = make_classification(n_samples=n_requests,n_features=10,random_state=42)
    
    latencies_ms = []
    
    for i in range(n_requests):
        single_input = X_test[i].reshape(1,-1)
        start = time.perf_counter()
        model.predict_proba(single_input)
        end = time.perf_counter()
        latencies_ms.append((end - start ) * 1000)
        
    latencies = np.array(latencies_ms)
    
    p50 = np.percentile(latencies,50)
    p95 = np.percentile(latencies,95)
    p99 = np.percentile(latencies,99)
    p999 = np.percentile(latencies,99.9)
    mean = latencies.mean()
    
    print(f"{'Percentile':<12} | {'Latency (ms)':>14} | {'SLO':>10} | Status")
    print("-" * 55)
    print(f"{'p50':<12} | {p50:>14.2f} | {'N/A':>10} | OK")
    print(f"{'p95':<12} | {p95:>14.2f} | {slo_p95_ms:>10} | {'PASS' if p95 < slo_p95_ms else 'FAIL'}")
    print(f"{'p99':<12} | {p99:>14.2f} | {slo_p99_ms:>10} | {'PASS' if p99 < slo_p99_ms else 'FAIL'}")
    print(f"{'p99.9':<12} | {p999:>14.2f} | {'N/A':>10} | INFO")
    print(f"{'mean':<12} | {mean:>14.2f} | {'N/A':>10} | INFO")
    
    passed = p95 < slo_p95_ms and p99 < slo_p99_ms
    
    print("\n" + "=" * 55)
    
    if passed:
        print("RESULT: PASS - Model meets latency SLOs")
        print("ACTION: Proceed with Kserve deployment")
    else:
        print("RESULT: FAIL - Model violates latency SLOs")
        print("ACTION: Block deployment. Investigate model size or complexity.")
        print("Suggestions:")
        
        if p95 >= slo_p95_ms:
            print(" - Reduce n_estimators in RandomForest")
            print(" - Consider model quantization")
            print(" - Try a simpler model type")
            
    return passed
X,y = make_classification(n_samples=5000,n_features=10, random_state=42)
X_train, _ ,y_train, _ =train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train,y_train)

import os
os.makedirs('models',exist_ok=True)
joblib.dump(model,'models/fraud_model.pk1')

passed = run_latency_slo_test(
    'models/fraud_model.pk1',
    n_requests=1000,
    slo_p95_ms=50,
    slo_p99_ms=100
)

sys.exit(0 if passed else 1)