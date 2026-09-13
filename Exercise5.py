import joblib
import json
import sys
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        'auc_roc': roc_auc_score(y_test, y_proba),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }


def promotion_gate(champion_path, challenger_path, X_test, y_test, thresholds, min_improvement=0.01):
    """ Decides whether challenger replaces champion
    Returns promote, hold or reject
    This runs inside GitHub Actions before any KServe deployments"""

    print("=" * 60)
    print("MODEL PROMOTION GATE")
    print("=" * 60)

    champ_metrics = evaluate_model(champion_path, X_test, y_test)
    chall_metrics = evaluate_model(challenger_path, X_test, y_test)

    print(f"\n{'Metric':<12} | {'Champion':>10} | {'Challenger':>12} | {'Delta':>8} | {'Status'}")
    print("-" * 65)

    gate_passed = True
    improvements = []

    # FIX: renamed loop variable from `thresholds` to `threshold_val` so it no
    # longer shadows the `thresholds` dict parameter passed into this function.
    for metric, threshold_val in thresholds.items():
        champ_val = champ_metrics[metric]
        chall_val = chall_metrics[metric]
        delta = chall_val - champ_val

        # Gate 1: Challenger must meet minimum absolute threshold
        meets_threshold = chall_val >= threshold_val

        # Gate 2: Challenger must not be significantly worse than champion.
        not_regressed = delta >= -0.005

        status = "PASS" if (meets_threshold and not_regressed) else "FAIL"
        if status == "FAIL":
            gate_passed = False

        improvements.append(delta)
        print(f"{metric:<12} | {champ_val:>10.4f} | {chall_val:>12.4f} | {delta:>+8.4f} | {status}")

    avg_improvement = np.mean(improvements)
    print(f"\nAverage Improvement: {avg_improvement:+.4f}")

    print("\n" + "=" * 60)

    if not gate_passed:
        print("RESULT: REJECT - Challenger failed minimum thresholds")
        print("ACTION: Keep Champion in Production. Alert Data Science Team")
        return "reject"

    if avg_improvement >= min_improvement:
        print("RESULT: PROMOTE - Challenger approved for deployment")
        print("ACTION: Trigger KServe canary rollout at 20% traffic")
        return "promote"

    print("RESULT: HOLD - Challenger meets thresholds but improvement minimal")
    print("ACTION: Keep Champion. Log Challenger for future comparison.")
    return "hold"


# Setup
X, y = make_classification(n_samples=5000, n_features=10, weights=[0.97, 0.03], random_state=42)

# FIX: added stratify=y so the 97/3 class ratio is preserved in both the
# train and test splits, rather than left to chance on a plain random split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs('models', exist_ok=True)

champion = RandomForestClassifier(n_estimators=50, random_state=42)
challenger = GradientBoostingClassifier(n_estimators=100, random_state=42)

champion.fit(X_train, y_train)
challenger.fit(X_train, y_train)

# FIX: corrected extension from .pk1 (digit one) to the conventional .pkl (letter L)
joblib.dump(champion, 'models/champion.pkl')
joblib.dump(challenger, 'models/challenger.pkl')

# Define your promotion thresholds.
# These are business requirements expressed as code
thresholds = {
    'auc_roc': 0.85,
    'recall': 0.70,
    'precision': 0.50,
}

decision = promotion_gate(
    'models/champion.pkl',
    'models/challenger.pkl',
    X_test,
    y_test,
    thresholds,
    min_improvement=0.005
)

# In GitHub Actions this becomes:
# if decision == "promote": trigger deployment
# if decision == "reject": fail the pipeline, notify the team
# if decision == "hold": pass pipeline, no deployment

exit_codes = {"promote": 0, "hold": 0, "reject": 1}
sys.exit(exit_codes[decision])