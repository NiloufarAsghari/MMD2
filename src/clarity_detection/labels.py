LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

BOUNDARY_LABELS = ["Ambivalent", "Clear Non-Reply"]
BOUNDARY_LABEL2ID = {label: idx for idx, label in enumerate(BOUNDARY_LABELS)}
BOUNDARY_ID2LABEL = {idx: label for label, idx in BOUNDARY_LABEL2ID.items()}

CLEAR_BOUNDARY_LABELS = ["Clear Reply", "Non-Clear"]
CLEAR_BOUNDARY_LABEL2ID = {label: idx for idx, label in enumerate(CLEAR_BOUNDARY_LABELS)}
CLEAR_BOUNDARY_ID2LABEL = {idx: label for label, idx in CLEAR_BOUNDARY_LABEL2ID.items()}

REPLY_BOUNDARY_LABELS = ["Clear Reply", "Ambivalent"]
REPLY_BOUNDARY_LABEL2ID = {label: idx for idx, label in enumerate(REPLY_BOUNDARY_LABELS)}
REPLY_BOUNDARY_ID2LABEL = {idx: label for label, idx in REPLY_BOUNDARY_LABEL2ID.items()}


def normalize_label(label: str) -> str:
    if label == "Ambiguous":
        return "Ambivalent"
    return label
