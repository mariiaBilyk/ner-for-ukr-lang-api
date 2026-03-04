import json
from sklearn.metrics import precision_recall_fscore_support
from src.utils.normalize_entities import normalize_entities

def ner_f1_score(pred, true):
    """
    Compute entity-level F1, precision, recall.
    pred: model output (string or list)
    true: ground truth (string or list)
    """

    true_set = normalize_entities(true)
    pred_set = normalize_entities(pred)

    # Convert sets into sorted lists for sklearn
    true_list = sorted(list(true_set))
    pred_list = sorted(list(pred_set))

    # If both empty → perfect score
    if len(true_list) == 0 and len(pred_list) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Prepare aligned lists
    # Labels
    all_labels = list({x[0] for x in true_list} | {x[0] for x in pred_list})

    # Convert entity tuples to strings for sklearn
    y_true = [f"{lbl}|{txt}" for (lbl, txt) in true_list]
    y_pred = [f"{lbl}|{txt}" for (lbl, txt) in pred_list]

    # Combine
    all_values = sorted(list(set(y_true) | set(y_pred)))

    # Create binary vectors
    y_true_bin = [1 if v in y_true else 0 for v in all_values]
    y_pred_bin = [1 if v in y_pred else 0 for v in all_values]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_dataset(df_true, df_pred):
    """
    df_true: dataframe with column 'entities'
    df_pred: dataframe with column 'entities'
    """

    all_true = []
    all_pred = []

    for i in range(len(df_true)):
        true_set = normalize_entities(df_true.iloc[i])
        pred_set = normalize_entities(df_pred.iloc[i])

        # Append individual (label,text) items globally
        for e in true_set:
            all_true.append(f"{e[0]}|{e[1]}")
        for e in pred_set:
            all_pred.append(f"{e[0]}|{e[1]}")

    # Ensure union of all unique entity strings
    all_values = sorted(list(set(all_true) | set(all_pred)))

    y_true = [1 if v in all_true else 0 for v in all_values]
    y_pred = [1 if v in all_pred else 0 for v in all_values]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    return {"precision": precision, "recall": recall, "f1": f1}
