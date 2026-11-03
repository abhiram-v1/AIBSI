from ml_app import node


@node(
    name="Site-normalized log-mel SVM",
    category="EEG",
    description="Train the validation-selected 20-PC class-balanced linear SVM and evaluate on held-out participants.",
    color="#aa733f",
    inputs=[{"name": "spectral", "type": "any"}],
    outputs=[{"name": "spectral_result", "type": "any"}],
)
def site_normalized_svm(spectral):
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    features = spectral["features"]
    labels = spectral["labels"]
    participants = spectral["participants"]
    splits = spectral["splits"]
    development = np.concatenate([splits["train"], splits["val"]])
    test = splits["test"]
    sites = np.asarray([p["site"] for p in participants])
    normalized = np.empty_like(features, dtype=np.float32)
    for site in sorted(set(sites)):
        reference = features[development[sites[development] == site]]
        if len(reference) == 0:
            raise ValueError("No development participants at site " + site)
        mean = reference.mean(axis=0, keepdims=True)
        std = np.maximum(reference.std(axis=0, keepdims=True), 1e-3)
        normalized[sites == site] = (features[sites == site] - mean) / std

    model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=20261009)),
        ("svm", SVC(C=0.1, kernel="linear", gamma="scale", class_weight="balanced", random_state=20261009)),
    ])
    model.fit(normalized[development], labels[development])
    prediction = model.predict(normalized[test])
    truth = labels[test]
    result = {
        "model": "site-normalized log-mel SVM",
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "correct": int(np.sum(truth == prediction)),
        "test_participants": int(len(test)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "settings": "moderate cleaning; 19 electrodes; 20 PCA components; linear SVM C=0.1; class balanced",
    }
    return {"spectral_result": result}
