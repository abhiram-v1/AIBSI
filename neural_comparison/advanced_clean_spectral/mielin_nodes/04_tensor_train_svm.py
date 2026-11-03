from ml_app import node


@node(
    name="Tensor Train + SVM",
    category="EEG",
    description="Fit a common training-only TT basis, then a class-balanced linear SVM using the prior CV-selected rank settings.",
    color="#8c61aa",
    inputs=[{"name": "tensor", "type": "any"}],
    outputs=[{"name": "tensor_result", "type": "any"}],
)
def tensor_train_svm(tensor):
    from scipy.linalg import eigh
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    values = tensor["tensors"]
    labels = tensor["labels"]
    participants = tensor["participants"]
    splits = tensor["splits"]
    development = np.concatenate([splits["train"], splits["val"]])
    test = splits["test"]
    sites = np.asarray([p["site"] for p in participants])
    normalized = np.empty_like(values, dtype=np.float32)
    for site in sorted(set(sites)):
        reference = values[development[sites[development] == site]].astype(np.float64)
        if len(reference) == 0:
            raise ValueError("No development participants at site " + site)
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), 1e-3)
        normalized[sites == site] = ((values[sites == site] - mean) / std).astype(np.float32)

    train_tensor = np.asarray(normalized[development], dtype=np.float64)
    tt_mean = train_tensor.mean(axis=0)
    dims = train_tensor.shape[1:]
    residual = np.moveaxis(train_tensor - tt_mean, 0, -1)
    cores = []
    last = 1
    for dim, rank in zip(dims, [4, 24, 4]):
        matrix = residual.reshape(last * dim, -1)
        rank = min(rank, *matrix.shape)
        covariance = matrix @ matrix.T
        _, vectors = eigh(covariance, subset_by_index=(len(covariance) - rank, len(covariance) - 1), check_finite=False)
        vectors = vectors[:, ::-1]
        signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])])
        vectors *= np.where(signs == 0, 1, signs)
        cores.append(vectors.reshape(last, dim, vectors.shape[1]))
        residual = vectors.T @ matrix
        last = vectors.shape[1]
    basis = np.einsum("aib,bjc,ckd->ijkd", *cores, optimize=True).reshape(np.prod(dims), last)
    features = (np.asarray(normalized) - tt_mean).reshape(len(normalized), -1) @ basis

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(C=0.1, kernel="linear", gamma="scale", class_weight="balanced", decision_function_shape="ovr", random_state=20261009)),
    ])
    model.fit(features[development], labels[development])
    prediction = model.predict(features[test])
    truth = labels[test]
    result = {
        "model": "Tensor Train + SVM",
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "correct": int(np.sum(truth == prediction)),
        "test_participants": int(len(test)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "tt_feature_count": int(features.shape[1]),
        "settings": "TT terminal rank 4; channel rank 4; spectral rank 24; linear SVM C=0.1; class balanced",
    }
    return {"tensor_result": result}
