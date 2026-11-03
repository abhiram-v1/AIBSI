from ml_app import node


@node(
    name="Held-out participant results",
    category="EEG",
    description="Show both model branches side by side on the same 27 held-out participants.",
    color="#457f82",
    inputs=[{"name": "spectral_result", "type": "any"}, {"name": "tensor_result", "type": "any"}],
    outputs=[{"name": "df", "type": "dataframe"}],
)
def compare_results(spectral_result, tensor_result):
    columns = ["model", "correct", "test_participants", "accuracy", "macro_f1", "macro_recall", "settings"]
    return {"df": pd.DataFrame([spectral_result, tensor_result])[columns]}
