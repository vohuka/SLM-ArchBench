import pandas as pd

def preprocess_archai_adr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing function for the ArchAI-ADR dataset.
    Assumes there are columns 'context' and 'decision'.
    It creates an instruction-style prompt and a target field.
    """
    src_col = "context"
    tgt_col = "decision"

    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(
            f"Expected columns '{src_col}' and '{tgt_col}' in ArchAI-ADR, "
            f"but got columns: {df.columns.tolist()}"
        )

    # Drop rows with missing source/target
    df = df.dropna(subset=[src_col, tgt_col]).copy()

    def build_prompt(row):
        return (
            "You are a software architecture assistant."
            "Given the following architecture decision context, write a clear and complete "
            "Architecture Decision Record (ADR) decision.\n\n"
            f"Context:\n{row[src_col]}\n\n"
            "Decision:"
        )

    df["prompt"] = df.apply(build_prompt, axis=1)
    df["target"] = df[tgt_col].astype(str)

    return df[["prompt", "target"]]
