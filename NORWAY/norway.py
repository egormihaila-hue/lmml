import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

tr = pd.read_csv("train-input.csv")
te = pd.read_csv("test-input.csv")
truth = pd.read_csv("train-output.csv").set_index("sample_id")
y_true = truth.loc[tr.sample_id, "corrected_label"].values

pix = [f"pixel_{i}" for i in range(64)]
Xtr = tr[pix].values / 16.0
Xte = te[pix].values / 16.0
g_tr = tr["given_label"].values
g_te = te["given_label"].values

def make_model():
    base_svc = SVC(C=10, gamma="scale", random_state=42)
    calibrated_svc = CalibratedClassifierCV(base_svc, ensemble=False)
    
    return VotingClassifier(
        [
            ("svm", calibrated_svc),
            ("knn", KNeighborsClassifier(3, weights="distance")),
            ("mlp", MLPClassifier((128,), max_iter=500, random_state=42)),
        ],
        voting="soft",
    )

print("Training base classifier...")
clf = make_model().fit(Xtr, y_true)
proba_te = clf.predict_proba(Xte)

def repair_nightwise(X, given, proba, block_size=120):

    fixed = given.copy()
    preds = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    
    n_samples = len(given)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        
        block_given = given[start:end]
        block_preds = preds[start:end]
        block_conf = conf[start:end]
        
        best_pair = None
        max_score = -1
        
        for a in range(10):
            for b in range(a + 1, 10):
                mask_a_to_b = (block_given == a) & (block_preds == b) & (block_conf > 0.6)
                mask_b_to_a = (block_given == b) & (block_preds == a) & (block_conf > 0.6)
                
                score = mask_a_to_b.sum() + mask_b_to_a.sum()
                if score > max_score and score >= 2:
                    max_score = score
                    best_pair = (a, b)
        
        if best_pair is not None:
            a, b = best_pair
            for j in range(len(block_given)):
                i = start + j
                if given[i] == a and preds[i] == b and conf[i] > 0.5:
                    fixed[i] = b
                elif given[i] == b and preds[i] == a and conf[i] > 0.5:
                    fixed[i] = a
        else:
            for j in range(len(block_given)):
                i = start + j
                if given[i] != preds[i] and conf[i] > 0.85:
                    fixed[i] = preds[i]
                    
    return fixed

print("Repairing labels in test dataset...")
fixed_te = repair_nightwise(Xte, g_te, proba_te, block_size=120)
print(f"Repaired samples in test: {(fixed_te != g_te).sum()} out of {len(g_te)}")

print("Performing final retraining...")
X_full = np.vstack([Xtr, Xte])
y_full = np.concatenate([y_true, fixed_te])

final_clf = make_model().fit(X_full, y_full)

proba_te2 = final_clf.predict_proba(Xte)
fixed_te2 = repair_nightwise(Xte, g_te, proba_te2, block_size=120)

pd.DataFrame({
    "sample_id": te.sample_id, 
    "corrected_label": fixed_te2
}).to_csv("test-output.csv", index=False)

joblib.dump(final_clf, "model.joblib")
print("Done: test-output.csv and model.joblib successfully created.")
