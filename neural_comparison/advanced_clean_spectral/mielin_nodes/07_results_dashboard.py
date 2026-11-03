from ml_app import node, Chart


@node(
    name="Training split + held-out dashboard",
    category="EEG",
    description="Show development/test cohort sizes, held-out model metrics, and three-class confusion matrices.",
    color="#6585b8",
    inputs=[
        {"name": "spectral", "type": "any"},
        {"name": "spectral_result", "type": "any"},
        {"name": "tensor_result", "type": "any"},
    ],
    outputs=[{"name": "graph_output", "type": "plot_spec"}],
)
def results_dashboard(spectral, spectral_result, tensor_result):
    splits = spectral["splits"]
    chart = Chart(rows=2, cols=2, title="EEG participant training and held-out evaluation")
    chart.subplot(0, 0, title="Participant split", y_label="Participants", legend=False).bar(
        ["Train", "Validation", "Test"],
        [len(splits["train"]), len(splits["val"]), len(splits["test"])],
        color="#568d62",
    )
    metrics = ["accuracy", "macro_f1", "macro_recall"]
    chart.subplot(0, 1, title="Held-out scores · 27 participants", y_label="Score", legend=False).bar(
        ["SVM acc", "SVM F1", "SVM recall", "TT acc", "TT F1", "TT recall"],
        [spectral_result[key] for key in metrics] + [tensor_result[key] for key in metrics],
        color="#4e90a6",
    )
    chart.subplot(1, 0, title="Log-mel SVM · actual rows, predicted columns", legend=False).heatmap(
        spectral_result["confusion_matrix"], labels=["AD", "FTD", "HC"],
        color_low="#e8f2fb", color_high="#367eb6",
    )
    chart.subplot(1, 1, title="Tensor Train + SVM · actual rows, predicted columns", legend=False).heatmap(
        tensor_result["confusion_matrix"], labels=["AD", "FTD", "HC"],
        color_low="#f2eafb", color_high="#8150a4",
    )
    return chart
