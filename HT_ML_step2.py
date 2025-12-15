import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import nltk
from transformers import pipeline
from pathlib import Path
import pandas as pd

nltk.download("stopwords", quiet=True)

def rtf_to_text(rtf: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+\d* ?", "", rtf)
    text = re.sub(r"[{}]", "", text)
    return text

def init_classifiers():
    """Initialize DeBERTa-v3 for intent and a pre-trained sentiment model."""
    print("Loading DeBERTa-v3 classifier...")
    deberta_clf = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        framework="pt",
        device=-1
    )

    print("Loading sentiment analysis model...")
    sentiment_clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        framework="pt",
        device=-1
    )

    return deberta_clf, sentiment_clf

def classify_intent(text: str, deberta_clf):
    candidate_labels = [
        "Team work",
        "Leadership",
        "High performer",
        "Social",
        "Caring",
        "Confident",
        "Humble",
        "Motivated",
        "Assertive",
        "Emotive",
        "Polite",
        "Redundant",
        "Formal",
        "Goal oriented",
        "People oriented",
        "Self-assured"
    ]

    out = deberta_clf(text, candidate_labels=candidate_labels, multi_label=True)
    return {lab: score for lab, score in zip(out["labels"], out["scores"])}

def classify_sentiment(text: str, sentiment_clf):
    """
    Returns positive, neutral, and negative scores.
    Note: this sentiment pipeline returns only the top label + score, not a full distribution.
    We approximate the other two by splitting the remainder.
    """
    pred = sentiment_clf(text, truncation=True, max_length=512)[0]
    label = pred["label"].lower()
    score = float(pred["score"])

    scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    if "positive" in label:
        scores["positive"] = score
        rem = (1 - score) / 2
        scores["neutral"] = rem
        scores["negative"] = rem
    elif "negative" in label:
        scores["negative"] = score
        rem = (1 - score) / 2
        scores["neutral"] = rem
        scores["positive"] = rem
    else:  # neutral (or anything else)
        scores["neutral"] = score
        rem = (1 - score) / 2
        scores["positive"] = rem
        scores["negative"] = rem

    return scores

BASE_DIR = Path("/Users/hollytaswell/Desktop/")

def process_csv_file(csv_path: Path, column_name: str, deberta_clf, sentiment_clf) -> pd.DataFrame | None:
    print(f"\n  Reading {csv_path.name}...")
    df_input = pd.read_csv(csv_path)

    if column_name not in df_input.columns:
        print(f"  Warning: Column '{column_name}' not found in {csv_path.name}")
        print(f"  Available columns: {', '.join(df_input.columns)}")
        return None

    paras = []
    for value in df_input[column_name]:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                paras.append(text)

    if not paras:
        print(f"  Warning: No valid paragraphs found in {csv_path.name}")
        return None

    print(f"  Found {len(paras)} paragraphs to analyze...")

    rows = []
    for i, paragraph in enumerate(paras, 1):
        if i % 10 == 0:
            print(f"    Processing paragraph {i}/{len(paras)}...")

        # FIXED: paragraph variable, valid assignment, and consistent cleaning
        clean_p = re.sub(r"[^\w\s]", " ", paragraph).lower()
        clean_p = re.sub(r"\s+", " ", clean_p).strip()

        intent_scores = classify_intent(clean_p, deberta_clf)
        intent_pct = {lbl: score * 100 for lbl, score in intent_scores.items()}  # FIXED: no leading space in keys

        sentiment_scores = classify_sentiment(clean_p, sentiment_clf)
        sentiment_pct = {k: v * 100 for k, v in sentiment_scores.items()}

        wc = len(paragraph.split())

        row = {
            "paragraph": paragraph,
            "word_count": wc,
            **sentiment_pct,
            **intent_pct,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Completed analysis of {csv_path.name}")
    return df

if __name__ == "__main__":
    mode = input("Process (1) single CSV file or (2) folder of CSV files? Enter 1 or 2: ").strip()

    if mode == "2":
        folder_input = input("Folder path containing CSV files: ").strip()
        p = Path(folder_input)
        folder_path = p if p.is_absolute() else BASE_DIR / p

        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Error: '{folder_path}' is not a valid folder.")
            raise SystemExit(1)

        csv_files = sorted(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in '{folder_path}'")
            raise SystemExit(1)

        print(f"\nFound {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")

        print("\nReading first CSV to determine columns...")
        first_df = pd.read_csv(csv_files[0])
        print("\nAvailable columns in CSVs:")
        for idx, col in enumerate(first_df.columns, 1):
            print(f"  {idx}. {col}")

        col_input = input("\nEnter column name or number to analyze: ").strip()
        if col_input.isdigit():
            col_idx = int(col_input) - 1
            column_name = first_df.columns[col_idx] if 0 <= col_idx < len(first_df.columns) else first_df.columns[0]
        elif col_input in first_df.columns:
            column_name = col_input
        else:
            column_name = first_df.columns[0]

        print(f"\nWill analyze column: {column_name}")

        out_name = input("Output Excel filename: ").strip()
        if not out_name.lower().endswith(".xlsx"):
            out_name += ".xlsx"
        o = Path(out_name)
        excel_out = o if o.is_absolute() else BASE_DIR / o

        print("\nInitializing classifiers...")
        deberta_clf, sentiment_clf = init_classifiers()

        results = []
        for csv_file in csv_files:
            sheet_name = csv_file.stem[:31]
            df_result = process_csv_file(csv_file, column_name, deberta_clf, sentiment_clf)
            if df_result is not None:
                results.append((sheet_name, df_result))

        if results:
            with pd.ExcelWriter(excel_out, engine="openpyxl", mode="w") as writer:
                for sheet_name, df_result in results:
                    df_result.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✓ Wrote sheet '{sheet_name}'")
            print(f"\n✓ Success! Created {excel_out} with {len(results)} sheets")
        else:
            print("\n✗ Error: No valid data to write. All CSV files were empty or had issues.")

    else:
        fname = input("CSV file name: ").strip()
        p = Path(fname)
        csv_path = p if p.is_absolute() else BASE_DIR / p

        print("Reading CSV file...")
        df_input = pd.read_csv(csv_path)

        print("\nAvailable columns:")
        for idx, col in enumerate(df_input.columns, 1):
            print(f"  {idx}. {col}")

        col_input = input("\nEnter column name or number to analyze: ").strip()
        if col_input.isdigit():
            col_idx = int(col_input) - 1
            column_name = df_input.columns[col_idx] if 0 <= col_idx < len(df_input.columns) else df_input.columns[0]
        elif col_input in df_input.columns:
            column_name = col_input
        else:
            column_name = df_input.columns[0]

        print(f"\nAnalyzing column: {column_name}")

        out_name = input("Output Excel filename: ").strip()
        if not out_name.lower().endswith(".xlsx"):
            out_name += ".xlsx"
        o = Path(out_name)
        excel_out = o if o.is_absolute() else BASE_DIR / o

        sheet_name = input("Sheet name: ").strip() or "Sheet1"

        print("\nInitializing classifiers...")
        deberta_clf, sentiment_clf = init_classifiers()

        df_result = process_csv_file(csv_path, column_name, deberta_clf, sentiment_clf)

        if df_result is not None:
            writer_mode = "a" if excel_out.exists() else "w"
            if writer_mode == "a":
                try:
                    with pd.ExcelWriter(excel_out, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
                        df_result.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                except Exception as e:
                    print(f"Error appending to Excel file: {e}")
                    print("Creating new file instead...")
                    with pd.ExcelWriter(excel_out, engine="openpyxl", mode="w") as writer:
                        df_result.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            else:
                with pd.ExcelWriter(excel_out, engine="openpyxl", mode="w") as writer:
                    df_result.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            print(f"\n✓ Success! Wrote sheet '{sheet_name[:31]}' to {excel_out}")
        else:
            print(f"\n✗ Error: No valid data to process from the CSV file.")
