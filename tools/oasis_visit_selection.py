"""Select actual recorded visits, without assuming consecutive MRI IDs."""
import re
import pandas as pd


def visit_ids(metadata, subject):
    frame = pd.read_excel(metadata)
    rows = frame[frame["Subject ID"] == subject].sort_values("MR Delay")
    identifiers = rows["MRI ID"].head(2).tolist()
    if len(identifiers) != 2 or len(set(identifiers)) != 2 or not all(re.fullmatch(re.escape(subject) + r"_MR\d+", x) for x in identifiers):
        raise ValueError("Two distinct, officially recorded MRI visits required")
    return identifiers
