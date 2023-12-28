import pandas as pd
from tqdm import tqdm
from pathlib import Path

SEED = 42
WEB_IMPORT_ROOT = Path("data/open-web-text-final")
GPT_IMPORT_ROOT = Path("data/open-gpt-text-final")
WEB_EXPORT_ROOT = Path("data/open-web-text-split")
GPT_EXPORT_ROOT = Path("data/open-gpt-text-split")

WEB_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
GPT_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

WEB_SUBSET = list(WEB_IMPORT_ROOT.glob("*.jsonl"))
WEB_TRAIN_DF = pd.DataFrame()
WEB_VALID_DF = pd.DataFrame()
WEB_TEST_DF = pd.DataFrame()

for subset in tqdm(WEB_SUBSET, desc="Processing Web"):
    df = pd.read_json(subset, lines=True)
    df = df.sample(frac=1, random_state=SEED)
    train_subset = df.iloc[:int(len(df) * 0.8)]
    WEB_TRAIN_DF = pd.concat([WEB_TRAIN_DF, train_subset], ignore_index=True)
    valid_subset = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    WEB_VALID_DF = pd.concat([WEB_VALID_DF, valid_subset], ignore_index=True)
    test_subset = df.iloc[int(len(df) * 0.9):]
    WEB_TEST_DF = pd.concat([WEB_TEST_DF, test_subset], ignore_index=True)

WEB_TRAIN_DF = WEB_TRAIN_DF.sample(frac=1, random_state=SEED)
WEB_VALID_DF = WEB_VALID_DF.sample(frac=1, random_state=SEED)
WEB_TEST_DF = WEB_TEST_DF.sample(frac=1, random_state=SEED)
WEB_TRAIN_DF.to_json(Path(WEB_EXPORT_ROOT, "train.jsonl"), orient="records", lines=True)
WEB_VALID_DF.to_json(Path(WEB_EXPORT_ROOT, "valid.jsonl"), orient="records", lines=True)
WEB_TEST_DF.to_json(Path(WEB_EXPORT_ROOT, "test.jsonl"), orient="records", lines=True)

GPT_SUBSET = list(GPT_IMPORT_ROOT.glob("*.jsonl"))
GPT_TRAIN_DF = pd.DataFrame()
GPT_VALID_DF = pd.DataFrame()
GPT_TEST_DF = pd.DataFrame()

for subset in tqdm(GPT_SUBSET, desc="Processing GPT"):
    df = pd.read_json(subset, lines=True)
    df = df.sample(frac=1, random_state=SEED)
    train_subset = df.iloc[:int(len(df) * 0.8)]
    GPT_TRAIN_DF = pd.concat([GPT_TRAIN_DF, train_subset], ignore_index=True)
    valid_subset = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    GPT_VALID_DF = pd.concat([GPT_VALID_DF, valid_subset], ignore_index=True)
    test_subset = df.iloc[int(len(df) * 0.9):]
    GPT_TEST_DF = pd.concat([GPT_TEST_DF, test_subset], ignore_index=True)

GPT_TRAIN_DF = GPT_TRAIN_DF.sample(frac=1, random_state=SEED)
GPT_VALID_DF = GPT_VALID_DF.sample(frac=1, random_state=SEED)
GPT_TEST_DF = GPT_TEST_DF.sample(frac=1, random_state=SEED)
GPT_TRAIN_DF.to_json(Path(GPT_EXPORT_ROOT, "train.jsonl"), orient="records", lines=True)
GPT_VALID_DF.to_json(Path(GPT_EXPORT_ROOT, "valid.jsonl"), orient="records", lines=True)
GPT_TEST_DF.to_json(Path(GPT_EXPORT_ROOT, "test.jsonl"), orient="records", lines=True)
