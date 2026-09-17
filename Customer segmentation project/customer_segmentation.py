"""
Customer Segmentation Analytics Dashboard
=========================================

A complete, single-file customer segmentation project built with Streamlit,
pandas, NumPy, scikit-learn, matplotlib and seaborn.

Install:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn

Run:
    streamlit run customer_segmentation.py

The application works immediately: if no CSV is uploaded, a realistic and
reproducible sample customer dataset is generated inside the program.
"""

from __future__ import annotations

import io
import re
import warnings

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DEFAULT_CLUSTERS = 4
PALETTE = "viridis"


# ---------------------------------------------------------------------------
# 1. COLUMN DEFINITIONS AND FLEXIBLE COLUMN DETECTION
# ---------------------------------------------------------------------------

# Canonical column name -> list of accepted alternative spellings.
COLUMN_ALIASES = {
    "CustomerID": [
        "customerid", "customer_id", "custid", "cust_id", "id", "clientid",
        "client_id", "customernumber", "customer_number", "userid", "user_id",
    ],
    "Age": ["age", "customerage", "customer_age", "ageyears", "age_years"],
    "Gender": ["gender", "sex", "customergender", "customer_gender"],
    "AnnualIncome": [
        "annualincome", "annual_income", "income", "yearlyincome",
        "yearly_income", "salary", "annualincomek", "annual_income_k",
        "incomeusd", "income_usd",
    ],
    "SpendingScore": [
        "spendingscore", "spending_score", "score", "spendscore",
        "spend_score", "spending", "spendingindex", "spending_index",
    ],
    "PurchaseFrequency": [
        "purchasefrequency", "purchase_frequency", "frequency", "freq",
        "purchasefreq", "purchase_freq", "ordersperyear", "orders_per_year",
        "numberofpurchases", "number_of_purchases",
    ],
    "AveragePurchaseValue": [
        "averagepurchasevalue", "avgpurchasevalue", "average_purchase_value",
        "avg_purchase_value", "averageordervalue", "avg_order_value",
        "averagespend", "avg_spend", "aov", "meanpurchasevalue",
    ],
    "TotalPurchases": [
        "totalpurchases", "total_purchases", "totalspend", "total_spend",
        "totalamount", "total_amount", "lifetimevalue", "lifetime_value",
        "ltv", "totalrevenue", "total_revenue",
    ],
    "WebsiteVisits": [
        "websitevisits", "website_visits", "visits", "sitevisits",
        "site_visits", "webvisits", "web_visits", "sessions", "pagevisits",
    ],
    "DiscountUsage": [
        "discountusage", "discount_usage", "discount", "discountrate",
        "discount_rate", "coupons", "couponusage", "coupon_usage",
        "discountpercent", "discount_percent",
    ],
    "ProductCategory": [
        "productcategory", "product_category", "category", "preferredcategory",
        "preferred_category", "favouritecategory", "favorite_category",
        "producttype", "product_type",
    ],
}

NUMERIC_CANONICAL = [
    "Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency",
    "AveragePurchaseValue", "TotalPurchases", "WebsiteVisits", "DiscountUsage",
]
CATEGORICAL_CANONICAL = ["Gender", "ProductCategory"]

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Groceries", "Home & Kitchen",
    "Beauty & Health", "Sports", "Books",
]


def normalize_name(name) -> str:
    """Lower-case a column name and strip everything that is not alphanumeric."""
    try:
        return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())
    except Exception:
        return ""


def detect_columns(df: pd.DataFrame) -> dict:
    """Map canonical column names to the actual columns found in ``df``."""
    mapping = {}
    used = set()
    if df is None or df.empty:
        return mapping

    normalized = {}
    for col in df.columns:
        normalized.setdefault(normalize_name(col), col)

    # Pass 1: exact (normalized) alias matches.
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            actual = normalized.get(alias)
            if actual is not None and actual not in used:
                mapping[canonical] = actual
                used.add(actual)
                break

    # Pass 2: partial matches for anything still missing.
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in mapping:
            continue
        for norm_col, actual in normalized.items():
            if actual in used or len(norm_col) < 4:
                continue
            if any(alias in norm_col or norm_col in alias for alias in aliases
                   if len(alias) > 3):
                mapping[canonical] = actual
                used.add(actual)
                break
    return mapping


# ---------------------------------------------------------------------------
# 2. SAMPLE DATA GENERATION (reproducible)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_sample_data(n_customers: int = 350, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic, reproducible sample customer dataset."""
    try:
        n_customers = int(n_customers)
    except Exception:
        n_customers = 350
    n_customers = int(np.clip(n_customers, 300, 5000))

    rng = np.random.default_rng(seed)

    # Four latent behavioural groups so clustering finds meaningful structure.
    group_specs = [
        # (share, age, income, spending, frequency, avg value, visits, discount)
        (0.22, (42, 9), (105_000, 18_000), (78, 10), (22, 5), (240, 55), (38, 9), (0.08, 0.04)),
        (0.33, (36, 10), (62_000, 12_000), (55, 12), (13, 4), (110, 30), (24, 7), (0.18, 0.07)),
        (0.27, (31, 9), (38_000, 9_000), (42, 13), (9, 3), (55, 18), (18, 6), (0.34, 0.10)),
        (0.18, (49, 12), (47_000, 14_000), (20, 9), (3, 2), (38, 15), (6, 3), (0.12, 0.08)),
    ]

    counts = [int(round(spec[0] * n_customers)) for spec in group_specs]
    counts[-1] = max(1, n_customers - sum(counts[:-1]))

    frames = []
    for group_id, (spec, count) in enumerate(zip(group_specs, counts)):
        if count <= 0:
            continue
        _, age, income, spend, freq, value, visits, discount = spec
        block = pd.DataFrame({
            "Age": rng.normal(age[0], age[1], count),
            "AnnualIncome": rng.normal(income[0], income[1], count),
            "SpendingScore": rng.normal(spend[0], spend[1], count),
            "PurchaseFrequency": rng.normal(freq[0], freq[1], count),
            "AveragePurchaseValue": rng.normal(value[0], value[1], count),
            "WebsiteVisits": rng.normal(visits[0], visits[1], count),
            "DiscountUsage": rng.normal(discount[0], discount[1], count),
            "_group": group_id,
        })
        frames.append(block)

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_rows = len(data)

    # Clip to realistic ranges.
    data["Age"] = np.clip(data["Age"], 18, 75).round(0)
    data["AnnualIncome"] = np.clip(data["AnnualIncome"], 12_000, 250_000).round(0)
    data["SpendingScore"] = np.clip(data["SpendingScore"], 1, 100).round(0)
    data["PurchaseFrequency"] = np.clip(data["PurchaseFrequency"], 0, 60).round(0)
    data["AveragePurchaseValue"] = np.clip(data["AveragePurchaseValue"], 10, 900).round(2)
    data["WebsiteVisits"] = np.clip(data["WebsiteVisits"], 0, 120).round(0)
    data["DiscountUsage"] = np.clip(data["DiscountUsage"], 0, 0.9).round(3)

    data["TotalPurchases"] = (
        data["PurchaseFrequency"] * data["AveragePurchaseValue"]
        * rng.normal(1.0, 0.08, n_rows)
    ).clip(lower=0).round(2)

    gender_pool = np.array(["Male", "Female", "Other"])
    data["Gender"] = rng.choice(gender_pool, size=n_rows, p=[0.47, 0.49, 0.04])

    category_probs = {
        0: [0.30, 0.16, 0.08, 0.18, 0.12, 0.10, 0.06],
        1: [0.18, 0.22, 0.16, 0.14, 0.13, 0.10, 0.07],
        2: [0.12, 0.26, 0.24, 0.10, 0.14, 0.08, 0.06],
        3: [0.10, 0.18, 0.30, 0.12, 0.14, 0.08, 0.08],
    }
    categories = []
    for group_id in data["_group"].to_numpy():
        probs = category_probs.get(int(group_id), [1 / len(PRODUCT_CATEGORIES)] * len(PRODUCT_CATEGORIES))
        categories.append(rng.choice(PRODUCT_CATEGORIES, p=probs))
    data["ProductCategory"] = categories

    data.insert(0, "CustomerID", [f"CUST{idx:05d}" for idx in range(1, n_rows + 1)])
    data = data.drop(columns=["_group"])

    ordered = [
        "CustomerID", "Age", "Gender", "AnnualIncome", "SpendingScore",
        "PurchaseFrequency", "AveragePurchaseValue", "TotalPurchases",
        "WebsiteVisits", "DiscountUsage", "ProductCategory",
    ]
    return data[ordered]


# ---------------------------------------------------------------------------
# 3. DATA LOADING, STANDARDIZING AND CLEANING
# ---------------------------------------------------------------------------

def read_uploaded_csv(uploaded_file):
    """Read an uploaded CSV defensively. Returns (dataframe_or_None, messages)."""
    messages = []
    if uploaded_file is None:
        return None, messages

    raw_bytes = b""
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
    except Exception as exc:
        messages.append(("error", f"The uploaded file could not be read ({exc})."))
        return None, messages

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        messages.append(("error", "The uploaded CSV file is empty."))
        return None, messages

    frame = None
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for separator in (None, ",", ";", "\t", "|"):
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    encoding=encoding,
                    sep=separator,
                    engine="python",
                )
            except Exception as exc:
                last_error = exc
                frame = None
                continue
            if frame is not None and frame.shape[1] >= 1 and not frame.empty:
                if encoding != "utf-8":
                    messages.append(("info", f"File decoded using '{encoding}' encoding."))
                return frame, messages
    messages.append((
        "error",
        f"The uploaded file could not be parsed as a valid CSV ({last_error}).",
    ))
    return None, messages


def standardize_dataframe(df: pd.DataFrame):
    """Rename recognised columns to canonical names and coerce numeric types."""
    messages = []
    if df is None or df.empty:
        return pd.DataFrame(), messages, {}

    work = df.copy()

    # Remove fully empty rows/columns and unnamed index columns.
    work = work.dropna(axis=1, how="all").dropna(axis=0, how="all")
    drop_cols = [c for c in work.columns if normalize_name(c).startswith("unnamed")]
    if drop_cols:
        work = work.drop(columns=drop_cols)

    if work.empty or work.shape[1] == 0:
        messages.append(("error", "The dataset contains no usable columns."))
        return pd.DataFrame(), messages, {}

    mapping = detect_columns(work)
    rename_map = {actual: canonical for canonical, actual in mapping.items()
                  if actual != canonical}
    if rename_map:
        work = work.rename(columns=rename_map)
        renamed_text = ", ".join(f"'{old}' → '{new}'" for old, new in rename_map.items())
        messages.append(("info", f"Columns automatically recognised: {renamed_text}."))

    # Remove duplicated column labels created by renaming.
    work = work.loc[:, ~work.columns.duplicated(keep="first")]

    # Coerce known numeric columns to numbers.
    for column in NUMERIC_CANONICAL:
        if column in work.columns:
            cleaned = work[column]
            if cleaned.dtype == object:
                cleaned = (
                    cleaned.astype(str)
                    .str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
                    .replace({"": np.nan, "-": np.nan, ".": np.nan})
                )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            bad = int(numeric.isna().sum() - work[column].isna().sum())
            if bad > 0:
                messages.append((
                    "warning",
                    f"{bad} non-numeric value(s) in '{column}' were treated as missing.",
                ))
            work[column] = numeric

    # Any remaining object column that is really numeric gets converted too.
    for column in work.columns:
        if column in NUMERIC_CANONICAL or column == "CustomerID":
            continue
        if work[column].dtype == object:
            converted = pd.to_numeric(work[column], errors="coerce")
            if converted.notna().mean() >= 0.85:
                work[column] = converted

    return work, messages, mapping


def clean_dataframe(df: pd.DataFrame):
    """Handle duplicates and missing values without ever raising."""
    messages = []
    if df is None or df.empty:
        return pd.DataFrame(), messages

    work = df.copy()

    if "CustomerID" in work.columns:
        duplicates = int(work["CustomerID"].duplicated().sum())
        if duplicates > 0:
            work = work.drop_duplicates(subset=["CustomerID"], keep="first")
            messages.append((
                "warning",
                f"{duplicates} duplicate CustomerID value(s) were removed.",
            ))
        work["CustomerID"] = work["CustomerID"].astype(str)
    else:
        work.insert(0, "CustomerID", [f"CUST{idx:05d}" for idx in range(1, len(work) + 1)])
        messages.append(("info", "No customer identifier found — IDs were generated automatically."))

    full_duplicates = int(work.duplicated().sum())
    if full_duplicates > 0:
        work = work.drop_duplicates(keep="first")
        messages.append(("warning", f"{full_duplicates} fully duplicated row(s) were removed."))

    numeric_cols = [c for c in work.select_dtypes(include=[np.number]).columns
                    if c != "CustomerID"]
    object_cols = [c for c in work.columns
                   if c not in numeric_cols and c != "CustomerID"]

    missing_numeric = 0
    for column in numeric_cols:
        missing = int(work[column].isna().sum())
        if missing > 0:
            missing_numeric += missing
            median = work[column].median()
            fill_value = 0.0 if pd.isna(median) else float(median)
            work[column] = work[column].fillna(fill_value)

    missing_text = 0
    for column in object_cols:
        missing = int(work[column].isna().sum())
        if missing > 0:
            missing_text += missing
            try:
                modes = work[column].mode(dropna=True)
                fill_value = modes.iloc[0] if len(modes) > 0 else "Unknown"
            except Exception:
                fill_value = "Unknown"
            work[column] = work[column].fillna(fill_value)
        work[column] = work[column].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})

    if missing_numeric or missing_text:
        messages.append((
            "info",
            f"Missing values handled: {missing_numeric} numeric (median fill) and "
            f"{missing_text} categorical (mode fill).",
        ))

    work = work.reset_index(drop=True)
    return work, messages


def select_feature_columns(df: pd.DataFrame):
    """Choose numeric and categorical feature columns that are safe to cluster on."""
    numeric_cols, categorical_cols = [], []
    if df is None or df.empty:
        return numeric_cols, categorical_cols

    for column in df.columns:
        if column in ("CustomerID", "Cluster", "Segment"):
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                continue
            if float(np.nanstd(values.to_numpy(dtype=float))) <= 1e-12:
                continue  # constant column: contributes nothing to clustering
            numeric_cols.append(column)
        else:
            unique_count = series.astype(str).nunique(dropna=True)
            if 2 <= unique_count <= 12:
                categorical_cols.append(column)
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# 4. PREPROCESSING AND CLUSTERING
# ---------------------------------------------------------------------------

def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """StandardScaler for numeric features, OneHotEncoder for categorical ones."""
    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, list(numeric_cols)))
    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ])
        transformers.append(("categorical", categorical_pipeline, list(categorical_cols)))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_feature_matrix(df: pd.DataFrame, numeric_cols, categorical_cols):
    """Return (matrix, error_message). ``matrix`` is None when preprocessing fails."""
    if df is None or df.empty:
        return None, "The dataset is empty."
    if not numeric_cols and not categorical_cols:
        return None, "No usable feature columns were found for clustering."

    try:
        subset = df[list(numeric_cols) + list(categorical_cols)].copy()
        for column in categorical_cols:
            subset[column] = subset[column].astype(str)
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        matrix = preprocessor.fit_transform(subset)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            return None, "There is not enough usable data to build a feature matrix."
        return matrix, None
    except Exception as exc:
        return None, f"Preprocessing failed: {exc}"


def max_allowed_clusters(n_samples: int) -> int:
    """Largest cluster count that is mathematically valid for the dataset."""
    try:
        n_samples = int(n_samples)
    except Exception:
        return 2
    return int(max(2, min(10, n_samples - 1)))


def fit_kmeans(matrix: np.ndarray, n_clusters: int):
    """Safely fit K-Means. Returns (model, labels, error_message)."""
    if matrix is None or getattr(matrix, "size", 0) == 0:
        return None, None, "No feature matrix is available for clustering."
    n_samples = int(matrix.shape[0])
    try:
        k = int(n_clusters)
    except Exception:
        k = DEFAULT_CLUSTERS
    k = int(np.clip(k, 2, max(2, n_samples)))
    if n_samples < 2:
        return None, None, "At least 2 customers are required for clustering."
    if k > n_samples:
        k = n_samples
    try:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10,
                       max_iter=300, random_state=RANDOM_STATE)
        labels = model.fit_predict(matrix)
        return model, np.asarray(labels), None
    except Exception as exc:
        return None, None, f"K-Means clustering failed: {exc}"


def safe_silhouette(matrix: np.ndarray, labels) -> float:
    """Silhouette score, or NaN when it is not mathematically valid."""
    try:
        if matrix is None or labels is None:
            return float("nan")
        labels = np.asarray(labels)
        n_samples = int(matrix.shape[0])
        unique = int(len(np.unique(labels)))
        if unique < 2 or unique >= n_samples:
            return float("nan")
        return float(silhouette_score(matrix, labels))
    except Exception:
        return float("nan")


@st.cache_data(show_spinner=False)
def evaluate_cluster_range(matrix: np.ndarray, k_max: int):
    """Compute inertia (elbow) and silhouette scores for k = 2..k_max."""
    ks, inertias, silhouettes = [], [], []
    if matrix is None or getattr(matrix, "size", 0) == 0:
        return ks, inertias, silhouettes

    n_samples = int(matrix.shape[0])
    k_max = int(min(max(2, k_max), max(2, n_samples - 1)))
    for k in range(2, k_max + 1):
        model, labels, error = fit_kmeans(matrix, k)
        if error is not None or model is None:
            continue
        ks.append(k)
        inertias.append(float(getattr(model, "inertia_", np.nan)))
        silhouettes.append(safe_silhouette(matrix, labels))
    return ks, inertias, silhouettes


def suggest_best_k(ks, silhouettes) -> int:
    """Recommend the k with the highest valid silhouette score."""
    best_k, best_score = DEFAULT_CLUSTERS, -np.inf
    for k, score in zip(ks, silhouettes):
        if score is None or not np.isfinite(score):
            continue
        if score > best_score:
            best_score, best_k = score, k
    return int(best_k) if np.isfinite(best_score) else DEFAULT_CLUSTERS


# ---------------------------------------------------------------------------
# 5. SEGMENT NAMING (data driven, never hardcoded to cluster 0)
# ---------------------------------------------------------------------------

VALUE_FEATURES = [
    "AnnualIncome", "SpendingScore", "PurchaseFrequency",
    "AveragePurchaseValue", "TotalPurchases",
]


def name_pool(k: int):
    """Ordered names from the most valuable segment to the least valuable one."""
    k = int(max(1, k))
    if k == 1:
        return ["All Customers"]
    if k == 2:
        return ["High Value Customers", "Budget Customers"]
    if k == 3:
        return ["High Value Customers", "Regular Customers",
                "At-Risk / Low Engagement Customers"]
    if k == 4:
        return ["High Value Customers", "Regular Customers", "Budget Customers",
                "At-Risk / Low Engagement Customers"]
    middle = [f"Mid-Tier Customers {i + 1}" for i in range(k - 5)]
    return (["High Value Customers", "Loyal High Spenders"] + middle
            + ["Regular Customers", "Budget Customers",
               "At-Risk / Low Engagement Customers"])


def assign_segment_names(df: pd.DataFrame, numeric_cols):
    """Rank clusters by an overall value score and attach meaningful names."""
    result = df.copy()
    if "Cluster" not in result.columns or result.empty:
        result["Segment"] = "Segment"
        return result, {}

    scoring_cols = [c for c in VALUE_FEATURES if c in result.columns
                    and pd.api.types.is_numeric_dtype(result[c])]
    if not scoring_cols:
        scoring_cols = [c for c in numeric_cols if c in result.columns
                        and pd.api.types.is_numeric_dtype(result[c])]

    cluster_ids = sorted(pd.unique(result["Cluster"]))
    if scoring_cols:
        try:
            means = result.groupby("Cluster")[scoring_cols].mean()
            std = means.std(axis=0, ddof=0).replace(0, np.nan)
            z_scores = (means - means.mean(axis=0)) / std
            z_scores = z_scores.fillna(0.0)
            composite = z_scores.mean(axis=1).sort_values(ascending=False)
            ordered_clusters = list(composite.index)
        except Exception:
            ordered_clusters = list(cluster_ids)
    else:
        ordered_clusters = list(cluster_ids)

    names = name_pool(len(ordered_clusters))
    mapping = {}
    for position, cluster_id in enumerate(ordered_clusters):
        mapping[cluster_id] = names[position] if position < len(names) else f"Segment {position + 1}"
    for cluster_id in cluster_ids:
        mapping.setdefault(cluster_id, f"Segment {cluster_id}")

    result["Segment"] = result["Cluster"].map(mapping).fillna("Unclassified")
    return result, mapping


# ---------------------------------------------------------------------------
# 6. SEGMENT STATISTICS AND DYNAMIC INSIGHTS
# ---------------------------------------------------------------------------

STAT_LABELS = [
    ("Age", "Average Age"),
    ("AnnualIncome", "Average Annual Income"),
    ("SpendingScore", "Average Spending Score"),
    ("PurchaseFrequency", "Average Purchase Frequency"),
    ("AveragePurchaseValue", "Average Purchase Value"),
    ("TotalPurchases", "Total Purchases"),
    ("WebsiteVisits", "Average Website Visits"),
    ("DiscountUsage", "Average Discount Usage"),
]


def safe_divide(numerator, denominator, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/NaN."""
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0 or not np.isfinite(denominator):
            return default
        value = numerator / denominator
        return float(value) if np.isfinite(value) else default
    except Exception:
        return default


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment statistics table."""
    if df is None or df.empty or "Segment" not in df.columns:
        return pd.DataFrame()

    total = len(df)
    rows = []
    try:
        grouped = df.groupby("Segment", dropna=False)
    except Exception:
        return pd.DataFrame()

    for segment, group in grouped:
        row = {
            "Segment": str(segment),
            "Customers": int(len(group)),
            "Percentage (%)": round(safe_divide(len(group) * 100.0, total), 2),
        }
        for column, label in STAT_LABELS:
            if column in group.columns and pd.api.types.is_numeric_dtype(group[column]):
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if values.empty:
                    continue
                if column == "TotalPurchases":
                    row[label] = round(float(values.sum()), 2)
                    row["Average Total Purchases"] = round(float(values.mean()), 2)
                else:
                    row[label] = round(float(values.mean()), 2)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty and "Customers" in summary.columns:
        summary = summary.sort_values("Customers", ascending=False).reset_index(drop=True)
    return summary


def describe_level(value, overall_mean, overall_std) -> str:
    """Classify a segment average as high / above average / average / low."""
    try:
        value = float(value)
        overall_mean = float(overall_mean)
        overall_std = float(overall_std)
    except Exception:
        return "typical"
    if not np.isfinite(value) or not np.isfinite(overall_mean):
        return "typical"
    if not np.isfinite(overall_std) or overall_std <= 1e-12:
        return "typical"
    z = (value - overall_mean) / overall_std
    if z >= 0.75:
        return "high"
    if z >= 0.25:
        return "above average"
    if z <= -0.75:
        return "low"
    if z <= -0.25:
        return "below average"
    return "typical"


def build_segment_insight(df: pd.DataFrame, segment: str) -> dict:
    """Generate a fully data-driven insight block for one segment."""
    insight = {
        "segment": str(segment),
        "size": 0,
        "share": 0.0,
        "characteristics": "Not enough data to describe this segment.",
        "behaviour": "Purchase behaviour could not be calculated.",
        "marketing": "Collect more data before designing a campaign.",
        "offers": "Standard offers.",
        "opportunity": "Undetermined.",
    }
    if df is None or df.empty or "Segment" not in df.columns:
        return insight

    group = df[df["Segment"] == segment]
    if group.empty:
        return insight

    insight["size"] = int(len(group))
    insight["share"] = round(safe_divide(len(group) * 100.0, len(df)), 2)

    levels = {}
    for column in ["AnnualIncome", "SpendingScore", "PurchaseFrequency",
                   "AveragePurchaseValue", "WebsiteVisits", "DiscountUsage", "Age"]:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            series = pd.to_numeric(df[column], errors="coerce")
            segment_values = pd.to_numeric(group[column], errors="coerce").dropna()
            if segment_values.empty:
                continue
            levels[column] = {
                "mean": float(segment_values.mean()),
                "level": describe_level(segment_values.mean(), series.mean(),
                                        series.std(ddof=0)),
            }

    def level_of(column: str) -> str:
        return levels.get(column, {}).get("level", "typical")

    def mean_of(column: str, default: float = float("nan")) -> float:
        return levels.get(column, {}).get("mean", default)

    income_level = level_of("AnnualIncome")
    spending_level = level_of("SpendingScore")
    frequency_level = level_of("PurchaseFrequency")
    value_level = level_of("AveragePurchaseValue")
    visits_level = level_of("WebsiteVisits")
    discount_level = level_of("DiscountUsage")

    parts = []
    if np.isfinite(mean_of("Age")):
        parts.append(f"an average age of about {mean_of('Age'):.0f} years")
    if np.isfinite(mean_of("AnnualIncome")):
        parts.append(f"{income_level} annual income (≈ {mean_of('AnnualIncome'):,.0f})")
    if np.isfinite(mean_of("SpendingScore")):
        parts.append(f"{spending_level} spending score (≈ {mean_of('SpendingScore'):.1f})")
    if "Gender" in group.columns:
        try:
            top_gender = group["Gender"].astype(str).mode()
            if len(top_gender) > 0:
                share = safe_divide((group["Gender"].astype(str) == top_gender.iloc[0]).sum() * 100.0,
                                    len(group))
                parts.append(f"mostly {top_gender.iloc[0]} customers ({share:.0f}%)")
        except Exception:
            pass
    insight["characteristics"] = (
        f"This segment covers {insight['size']} customers ({insight['share']:.1f}% of the base) with "
        + ", ".join(parts) + "." if parts else
        f"This segment covers {insight['size']} customers ({insight['share']:.1f}% of the base)."
    )

    behaviour_parts = []
    if np.isfinite(mean_of("PurchaseFrequency")):
        behaviour_parts.append(
            f"{frequency_level} purchase frequency (≈ {mean_of('PurchaseFrequency'):.1f} purchases)")
    if np.isfinite(mean_of("AveragePurchaseValue")):
        behaviour_parts.append(
            f"{value_level} average order value (≈ {mean_of('AveragePurchaseValue'):,.2f})")
    if np.isfinite(mean_of("WebsiteVisits")):
        behaviour_parts.append(f"{visits_level} website engagement")
    if np.isfinite(mean_of("DiscountUsage")):
        behaviour_parts.append(f"{discount_level} reliance on discounts")
    if "ProductCategory" in group.columns:
        try:
            top_category = group["ProductCategory"].astype(str).mode()
            if len(top_category) > 0:
                behaviour_parts.append(f"a preference for {top_category.iloc[0]}")
        except Exception:
            pass
    insight["behaviour"] = ("They show " + ", ".join(behaviour_parts) + "."
                            if behaviour_parts else "Purchase behaviour is close to the overall average.")

    high_levels = {"high", "above average"}
    low_levels = {"low", "below average"}
    premium = income_level in high_levels and spending_level in high_levels
    loyal = frequency_level in high_levels
    price_sensitive = discount_level in high_levels or income_level in low_levels
    disengaged = (spending_level == "low" and frequency_level == "low"
                  and discount_level not in high_levels)

    if premium and loyal:
        insight["marketing"] = (
            "Treat this group as the priority tier: personalised premium recommendations, "
            "early access to new arrivals and a VIP loyalty programme with a dedicated relationship manager.")
        insight["offers"] = ("Exclusive memberships, bundled premium products, free express delivery "
                             "and points-based rewards instead of deep discounts.")
        insight["opportunity"] = ("Highest revenue contribution per customer — protect retention here, "
                                  "because losing a single customer is expensive.")
    elif premium:
        insight["marketing"] = ("Encourage more frequent visits with personalised product suggestions "
                                "and reminder/replenishment campaigns.")
        insight["offers"] = "Premium bundles, limited-time upgrades and cross-category recommendations."
        insight["opportunity"] = "High spending power that is not yet converted into frequent purchases."
    elif loyal and value_level in low_levels:
        insight["marketing"] = ("Focus on basket-size growth: cross-sell, upsell and 'frequently bought "
                                "together' suggestions at checkout.")
        insight["offers"] = "Volume discounts, free shipping above a threshold and combo packs."
        insight["opportunity"] = "Frequent buyers whose average order value can realistically be increased."
    elif disengaged:
        insight["marketing"] = ("Run win-back and re-engagement campaigns: reminder emails, short surveys "
                                "to understand churn reasons and time-limited comeback offers.")
        insight["offers"] = "Welcome-back vouchers, first-purchase-of-the-quarter discounts and free trials."
        insight["opportunity"] = ("Churn risk — a small reactivation rate still adds measurable revenue, "
                                  "but keep acquisition spend low for this group.")
    elif price_sensitive:
        insight["marketing"] = ("Value-first messaging: highlight savings, seasonal sales and clear "
                                "price comparisons across affordable products.")
        insight["offers"] = "Coupon codes, festival sales, cashback and entry-level product ranges."
        insight["opportunity"] = ("Large but margin-sensitive group — grow volume with low-cost campaigns "
                                  "rather than expensive personalisation.")
    else:
        insight["marketing"] = ("Steady nurturing with newsletters, category recommendations and "
                                "loyalty-point reminders to move them into a higher value tier.")
        insight["offers"] = "Moderate seasonal discounts, referral bonuses and loyalty milestones."
        insight["opportunity"] = ("The core of the customer base — small improvements in frequency or "
                                  "order value create a large aggregate gain.")
    return insight


# ---------------------------------------------------------------------------
# 7. PLOTTING HELPERS (all defensive)
# ---------------------------------------------------------------------------

def show_figure(fig):
    """Render a matplotlib figure in Streamlit and free the memory."""
    try:
        st.pyplot(fig)
    except Exception as exc:
        st.warning(f"This chart could not be displayed ({exc}).")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def plot_elbow(ks, inertias):
    if not ks or not inertias:
        st.info("The elbow curve needs at least two valid cluster counts.")
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(ks, inertias, marker="o", color="#2b6cb0", linewidth=2,
                label="Within-cluster sum of squares (inertia)")
        ax.set_title("Elbow Method — Optimal Number of Clusters", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia (WCSS)")
        ax.set_xticks(ks)
        ax.legend(loc="best")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Elbow curve could not be plotted ({exc}).")


def plot_silhouette(ks, silhouettes):
    valid = [(k, s) for k, s in zip(ks, silhouettes) if s is not None and np.isfinite(s)]
    if not valid:
        st.info("Silhouette scores are not mathematically valid for this dataset.")
        return
    try:
        xs = [k for k, _ in valid]
        ys = [s for _, s in valid]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar([str(x) for x in xs], ys, color=sns.color_palette(PALETTE, len(xs)),
                      label="Silhouette score")
        best_index = int(np.argmax(ys))
        bars[best_index].set_edgecolor("black")
        bars[best_index].set_linewidth(2)
        for bar, value in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Silhouette Score Comparison (higher is better)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_ylim(min(0.0, min(ys) - 0.05), max(ys) + 0.08)
        ax.legend(loc="best")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Silhouette chart could not be plotted ({exc}).")


def plot_segment_distribution(df: pd.DataFrame):
    if df is None or df.empty or "Segment" not in df.columns:
        st.info("No segment data available for this chart.")
        return
    try:
        counts = df["Segment"].value_counts()
        if counts.empty:
            st.info("No customers available in the current selection.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        colors = sns.color_palette(PALETTE, len(counts))
        axes[0].bar(counts.index.astype(str), counts.values, color=colors,
                    label="Customers per segment")
        axes[0].set_title("Customer Segment Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Segment")
        axes[0].set_ylabel("Number of customers")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(loc="best", fontsize=8)
        for index, value in enumerate(counts.values):
            axes[0].text(index, value, str(int(value)), ha="center", va="bottom", fontsize=9)

        axes[1].pie(counts.values, labels=[str(i) for i in counts.index], autopct="%1.1f%%",
                    colors=colors, startangle=110,
                    wedgeprops={"edgecolor": "white", "linewidth": 1})
        axes[1].set_title("Segment Share of Customer Base", fontsize=12, fontweight="bold")
        axes[1].axis("equal")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Segment distribution chart failed ({exc}).")


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    if df is None or df.empty:
        st.info("No customers available in the current selection.")
        return
    if x_col not in df.columns or y_col not in df.columns:
        st.info(f"Columns '{x_col}' and '{y_col}' are not available in this dataset.")
        return
    try:
        plot_df = df[[x_col, y_col, "Segment"]].copy() if "Segment" in df.columns \
            else df[[x_col, y_col]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col, y_col])
        if plot_df.empty:
            st.info("Not enough numeric data to draw this scatter plot.")
            return
        fig, ax = plt.subplots(figsize=(8.5, 5))
        if "Segment" in plot_df.columns:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue="Segment",
                            palette=PALETTE, s=60, alpha=0.85, edgecolor="white", ax=ax)
            ax.legend(title="Segment", loc="best", fontsize=8, title_fontsize=9)
        else:
            ax.scatter(plot_df[x_col], plot_df[y_col], s=60, alpha=0.85,
                       color="#2b6cb0", label="Customers")
            ax.legend(loc="best")
        ax.set_title(f"{x_col} vs {y_col} by Segment", fontsize=13, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Scatter plot failed ({exc}).")


def plot_distribution_by_segment(df: pd.DataFrame, column: str, kind: str = "box"):
    if df is None or df.empty or column not in df.columns or "Segment" not in df.columns:
        st.info(f"'{column}' is not available for this chart.")
        return
    try:
        plot_df = df[[column, "Segment"]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.dropna(subset=[column])
        if plot_df.empty:
            st.info(f"No numeric values available in '{column}'.")
            return
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if kind == "hist":
            for color, (segment, group) in zip(
                    sns.color_palette(PALETTE, plot_df["Segment"].nunique()),
                    plot_df.groupby("Segment")):
                ax.hist(group[column], bins=15, alpha=0.6, label=str(segment), color=color)
            ax.set_ylabel("Number of customers")
            ax.set_xlabel(column)
            ax.legend(title="Segment", fontsize=8, title_fontsize=9)
        elif kind == "violin":
            sns.violinplot(data=plot_df, x="Segment", y=column, hue="Segment",
                           palette=PALETTE, legend=False, ax=ax)
            ax.set_xlabel("Segment")
            ax.set_ylabel(column)
            ax.tick_params(axis="x", rotation=20)
        else:
            sns.boxplot(data=plot_df, x="Segment", y=column, hue="Segment",
                        palette=PALETTE, legend=False, ax=ax)
            ax.set_xlabel("Segment")
            ax.set_ylabel(column)
            ax.tick_params(axis="x", rotation=20)
        ax.set_title(f"{column} by Customer Segment", fontsize=13, fontweight="bold")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Distribution chart for '{column}' failed ({exc}).")


def plot_mean_bar(df: pd.DataFrame, column: str, title: str):
    if df is None or df.empty or column not in df.columns or "Segment" not in df.columns:
        st.info(f"'{column}' is not available for this chart.")
        return
    try:
        values = pd.to_numeric(df[column], errors="coerce")
        temp = pd.DataFrame({"Segment": df["Segment"], column: values}).dropna()
        if temp.empty:
            st.info(f"No numeric values available in '{column}'.")
            return
        means = temp.groupby("Segment")[column].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        bars = ax.bar(means.index.astype(str), means.values,
                      color=sns.color_palette(PALETTE, len(means)),
                      label=f"Average {column}")
        for bar, value in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:,.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Segment")
        ax.set_ylabel(f"Average {column}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Bar chart for '{column}' failed ({exc}).")


def plot_segment_comparison(df: pd.DataFrame, numeric_cols):
    if df is None or df.empty or "Segment" not in df.columns:
        st.info("Segment comparison is not available for this selection.")
        return
    usable = [c for c in numeric_cols if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])][:6]
    if not usable:
        st.info("No numeric features available for the comparison chart.")
        return
    try:
        means = df.groupby("Segment")[usable].mean()
        if means.empty:
            st.info("No data available for the comparison chart.")
            return
        span = (means.max() - means.min()).replace(0, np.nan)
        normalized = ((means - means.min()) / span).fillna(0.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        n_segments = len(normalized.index)
        width = 0.8 / max(1, n_segments)
        positions = np.arange(len(usable))
        colors = sns.color_palette(PALETTE, n_segments)
        for index, (segment, row) in enumerate(normalized.iterrows()):
            ax.bar(positions + index * width, row.values, width=width,
                   label=str(segment), color=colors[index])
        ax.set_xticks(positions + width * (n_segments - 1) / 2)
        ax.set_xticklabels(usable, rotation=18, ha="right")
        ax.set_title("Segment Comparison Across Key Features (min-max normalised)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Normalised average (0 = lowest, 1 = highest)")
        ax.set_ylim(0, 1.15)
        ax.legend(title="Segment", fontsize=8, title_fontsize=9, loc="upper right")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Segment comparison chart failed ({exc}).")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols):
    usable = [c for c in numeric_cols if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])]
    if df is None or df.empty or len(usable) < 2:
        st.info("At least two numeric columns are required for a correlation heatmap.")
        return
    try:
        correlation = df[usable].corr(numeric_only=True)
        correlation = correlation.dropna(how="all").dropna(axis=1, how="all")
        if correlation.empty or correlation.shape[0] < 2:
            st.info("Correlations could not be computed for this dataset.")
            return
        fig, ax = plt.subplots(figsize=(min(11, 1.35 * len(correlation) + 3),
                                        min(9, 1.1 * len(correlation) + 2.5)))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.5, square=False, cbar_kws={"label": "Correlation"}, ax=ax)
        ax.set_title("Correlation Heatmap of Numerical Features",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Correlation heatmap failed ({exc}).")


def plot_pca(matrix: np.ndarray, labels, segment_names):
    if matrix is None or getattr(matrix, "size", 0) == 0:
        st.info("PCA needs a valid feature matrix.")
        return
    try:
        n_samples, n_features = int(matrix.shape[0]), int(matrix.shape[1])
        if n_samples < 2 or n_features < 2:
            st.info("PCA requires at least 2 samples and 2 features.")
            return
        components = int(min(2, n_features, n_samples))
        pca = PCA(n_components=components, random_state=RANDOM_STATE)
        reduced = pca.fit_transform(matrix)
        if reduced.shape[1] < 2:
            reduced = np.column_stack([reduced[:, 0], np.zeros(len(reduced))])
            explained = [float(pca.explained_variance_ratio_[0]), 0.0]
        else:
            explained = [float(v) for v in pca.explained_variance_ratio_[:2]]

        plot_df = pd.DataFrame({"PC1": reduced[:, 0], "PC2": reduced[:, 1]})
        if segment_names is not None and len(segment_names) == len(plot_df):
            plot_df["Segment"] = [str(s) for s in segment_names]
        elif labels is not None and len(labels) == len(plot_df):
            plot_df["Segment"] = [f"Cluster {int(l)}" for l in labels]
        else:
            plot_df["Segment"] = "All customers"

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Segment", palette=PALETTE,
                        s=65, alpha=0.85, edgecolor="white", ax=ax)
        ax.set_title("2D PCA Projection of Customer Segments", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Principal Component 1 ({explained[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"Principal Component 2 ({explained[1] * 100:.1f}% variance)")
        ax.legend(title="Segment", fontsize=8, title_fontsize=9, loc="best")
        fig.tight_layout()
        show_figure(fig)
        st.caption(
            f"The two components together explain "
            f"{(explained[0] + explained[1]) * 100:.1f}% of the total variance."
        )
    except Exception as exc:
        st.warning(f"PCA visualisation failed ({exc}).")


def plot_category_by_segment(df: pd.DataFrame, column: str):
    if df is None or df.empty or column not in df.columns or "Segment" not in df.columns:
        return
    try:
        table = pd.crosstab(df["Segment"].astype(str), df[column].astype(str))
        if table.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        table.plot(kind="bar", stacked=True, ax=ax,
                   color=sns.color_palette(PALETTE, table.shape[1]))
        ax.set_title(f"{column} Mix by Customer Segment", fontsize=13, fontweight="bold")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Number of customers")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title=column, fontsize=8, title_fontsize=9,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        show_figure(fig)
    except Exception as exc:
        st.warning(f"Category chart for '{column}' failed ({exc}).")


# ---------------------------------------------------------------------------
# 8. DASHBOARD SECTIONS
# ---------------------------------------------------------------------------

def render_kpis(df: pd.DataFrame, n_segments: int):
    """KPI metric cards."""
    try:
        columns = st.columns(5)
        columns[0].metric("Total Customers", f"{len(df):,}")
        columns[1].metric("Number of Segments", f"{int(n_segments)}")

        def metric_value(column, fmt, default="N/A"):
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if not values.empty:
                    return fmt.format(float(values.mean()))
            return default

        columns[2].metric("Average Income", metric_value("AnnualIncome", "{:,.0f}"))
        columns[3].metric("Avg Spending Score", metric_value("SpendingScore", "{:.1f}"))
        columns[4].metric("Avg Purchase Value", metric_value("AveragePurchaseValue", "{:,.2f}"))
    except Exception as exc:
        st.warning(f"KPI cards could not be rendered ({exc}).")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters for gender, age, income and segment."""
    filtered = df.copy()
    st.sidebar.markdown("### 🔍 Customer Filters")

    try:
        if "Segment" in filtered.columns:
            options = sorted(filtered["Segment"].astype(str).unique().tolist())
            chosen = st.sidebar.multiselect("Segment", options, default=options)
            if chosen:
                filtered = filtered[filtered["Segment"].astype(str).isin(chosen)]

        if "Gender" in filtered.columns:
            options = sorted(df["Gender"].astype(str).unique().tolist())
            if 1 <= len(options) <= 20:
                chosen = st.sidebar.multiselect("Gender", options, default=options)
                if chosen:
                    filtered = filtered[filtered["Gender"].astype(str).isin(chosen)]

        for column, label in (("Age", "Age range"), ("AnnualIncome", "Annual income range")):
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if values.empty:
                    continue
                low, high = float(values.min()), float(values.max())
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    continue
                step = max((high - low) / 100.0, 0.01)
                selected = st.sidebar.slider(label, min_value=float(low), max_value=float(high),
                                             value=(float(low), float(high)), step=float(step))
                series = pd.to_numeric(filtered[column], errors="coerce")
                filtered = filtered[series.between(selected[0], selected[1], inclusive="both")
                                    | series.isna()]
    except Exception as exc:
        st.sidebar.warning(f"Filters could not be applied ({exc}).")
        return df

    return filtered


def render_about_section():
    st.subheader("📘 About This Project")
    st.markdown(
        """
**Problem statement** — Businesses treat every customer the same way, which wastes marketing
budget on people who will never convert while under-serving the customers who generate most of
the revenue. Without segmentation there is no way to know which group deserves which campaign.

**Objective** — Segment customers based on their demographics (age, gender, income) and their
behaviour (spending score, purchase frequency, order value, website visits, discount usage), and
turn each segment into a concrete marketing recommendation.

**Dataset** — The app accepts any customer CSV upload. When no file is uploaded it generates a
reproducible sample of 350+ customers (fixed random seed) with realistic demographic and
behavioural columns, so the project always runs.

**Data preprocessing** — Recognised columns are renamed to standard names, non-numeric entries in
numeric columns are converted, missing numeric values are filled with the median, missing
categorical values with the mode, duplicate customer IDs are removed, and constant columns are
excluded. Numerical features are scaled with `StandardScaler` and categorical features encoded
with `OneHotEncoder` inside a `ColumnTransformer` pipeline.

**K-Means clustering** — K-Means partitions customers into *k* groups by minimising the distance
between each customer and its cluster centre. Scaling matters because K-Means uses Euclidean
distance, so an unscaled income column would dominate a 1–100 spending score.

**Elbow method** — Inertia (within-cluster sum of squares) is plotted for a range of *k*. The
"elbow", where the curve stops dropping sharply, indicates a reasonable cluster count.

**Silhouette score** — Measures how similar a customer is to its own cluster compared to the
nearest other cluster, ranging from -1 to 1. Higher is better; the app recommends the *k* with
the highest score and only computes it when it is mathematically valid.

**PCA** — Principal Component Analysis compresses all scaled features into two components so the
segments can be inspected visually on a 2D plot.

**Business insights** — Every segment is ranked on a composite value score, named from its actual
statistics (never from a hardcoded cluster number) and paired with an automatically generated
marketing approach, offer suggestion and business opportunity.
        """
    )


# ---------------------------------------------------------------------------
# 9. MAIN APPLICATION
# ---------------------------------------------------------------------------

def main():
    try:
        st.set_page_config(
            page_title="Customer Segmentation Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    st.title("📊 Customer Segmentation Analytics Dashboard")
    st.markdown(
        "**Analyze customer behavior, demographics and purchasing patterns using "
        "K-Means clustering.**"
    )
    st.divider()

    # ---------------- Sidebar: data source ----------------
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("### 📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a customer CSV file (optional)", type=["csv"],
        help="If no file is uploaded, a realistic sample dataset is generated automatically.",
    )
    sample_size = st.sidebar.slider("Sample dataset size (used if no CSV is uploaded)",
                                    min_value=300, max_value=1500, value=350, step=50)

    raw_df, data_source = None, "sample"
    startup_messages = []

    if uploaded_file is not None:
        candidate, messages = read_uploaded_csv(uploaded_file)
        startup_messages.extend(messages)
        if candidate is not None and not candidate.empty:
            raw_df, data_source = candidate, "upload"
        else:
            startup_messages.append(
                ("warning", "Falling back to the built-in sample dataset."))

    if raw_df is None:
        try:
            raw_df = generate_sample_data(sample_size)
        except Exception as exc:
            st.error(f"The sample dataset could not be generated ({exc}).")
            st.stop()

    # ---------------- Preprocessing ----------------
    standardized, std_messages, detected = standardize_dataframe(raw_df)
    startup_messages.extend(std_messages)

    if standardized is None or standardized.empty:
        startup_messages.append(
            ("error", "The uploaded dataset had no usable content — using the sample dataset."))
        raw_df = generate_sample_data(sample_size)
        standardized, std_messages, detected = standardize_dataframe(raw_df)
        startup_messages.extend(std_messages)
        data_source = "sample"

    cleaned, clean_messages = clean_dataframe(standardized)
    startup_messages.extend(clean_messages)

    numeric_cols, categorical_cols = select_feature_columns(cleaned)

    # Guard: too few rows or not enough numeric information.
    fallback_needed = False
    if len(cleaned) < 5:
        startup_messages.append(
            ("error", "The dataset has fewer than 5 usable rows, which is too few for clustering."))
        fallback_needed = True
    elif len(numeric_cols) < 2:
        startup_messages.append((
            "error",
            "The dataset does not contain at least two usable numerical columns "
            "(constant and non-numeric columns are ignored).",
        ))
        fallback_needed = True

    if fallback_needed and data_source == "upload":
        startup_messages.append(
            ("warning", "Switching to the built-in sample dataset so the analysis can continue."))
        raw_df = generate_sample_data(sample_size)
        standardized, _, detected = standardize_dataframe(raw_df)
        cleaned, _ = clean_dataframe(standardized)
        numeric_cols, categorical_cols = select_feature_columns(cleaned)
        data_source = "sample"
    elif fallback_needed:
        st.error("Clustering cannot be performed on this dataset.")
        st.stop()

    with st.sidebar.expander("ℹ️ Data status", expanded=False):
        st.write(f"**Source:** {'Uploaded CSV' if data_source == 'upload' else 'Generated sample data'}")
        st.write(f"**Rows:** {len(cleaned):,}")
        st.write(f"**Numeric features used:** {len(numeric_cols)}")
        st.write(f"**Categorical features used:** {len(categorical_cols)}")

    # ---------------- Feature matrix ----------------
    matrix, matrix_error = build_feature_matrix(cleaned, numeric_cols, categorical_cols)
    if matrix is None:
        st.error(matrix_error or "The feature matrix could not be built.")
        st.stop()

    # ---------------- Cluster evaluation ----------------
    k_limit = max_allowed_clusters(len(cleaned))
    try:
        ks, inertias, silhouettes = evaluate_cluster_range(matrix, k_limit)
    except Exception:
        ks, inertias, silhouettes = [], [], []
    recommended_k = suggest_best_k(ks, silhouettes)

    st.sidebar.markdown("### 🎯 Clustering Settings")
    default_k = int(np.clip(DEFAULT_CLUSTERS, 2, k_limit))
    n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=int(k_limit),
                                   value=default_k, step=1)
    st.sidebar.caption(
        f"Recommended by silhouette score: **k = {recommended_k}** "
        f"(maximum allowed here: {k_limit})."
    )
    if st.sidebar.checkbox("Use the recommended number of clusters", value=False):
        n_clusters = int(np.clip(recommended_k, 2, k_limit))

    # ---------------- Fit the model ----------------
    model, labels, cluster_error = fit_kmeans(matrix, n_clusters)
    if cluster_error is not None or labels is None:
        st.error(cluster_error or "Clustering failed unexpectedly.")
        st.stop()

    segmented = cleaned.copy()
    segmented["Cluster"] = labels.astype(int)
    segmented, name_mapping = assign_segment_names(segmented, numeric_cols)
    final_silhouette = safe_silhouette(matrix, labels)
    n_segments = int(segmented["Segment"].nunique())

    for level, message in startup_messages:
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)

    if data_source == "upload":
        st.success(f"Uploaded dataset loaded successfully — {len(segmented):,} customers analysed.")
    else:
        st.info(f"No CSV uploaded, so a generated sample of {len(segmented):,} customers is being analysed.")

    # ---------------- Filters ----------------
    filtered = apply_filters(segmented)
    if filtered is None or filtered.empty:
        st.warning("No customers match the selected filters. Showing the full dataset instead.")
        filtered = segmented.copy()

    render_kpis(filtered, n_segments)
    score_text = f"{final_silhouette:.3f}" if np.isfinite(final_silhouette) else "not available"
    st.caption(
        f"Model: K-Means with k = {n_clusters} · Silhouette score = {score_text} · "
        f"Features used: {len(numeric_cols)} numerical, {len(categorical_cols)} categorical."
    )
    st.divider()

    tabs = st.tabs([
        "📋 Data Overview",
        "🔎 Cluster Selection",
        "📈 Visual Analysis",
        "🧩 Segment Analysis",
        "💡 Targeted Insights",
        "⬇️ Download & About",
    ])

    # ---------------- Tab 1: data overview ----------------
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.markdown(
            "The table below shows the dataset after preprocessing (column detection, type "
            "conversion, duplicate removal and missing-value handling)."
        )
        try:
            st.dataframe(filtered.head(200), width="stretch")
        except Exception:
            try:
                st.dataframe(filtered.head(200))
            except Exception as exc:
                st.warning(f"The data table could not be displayed ({exc}).")

        left, right = st.columns(2)
        with left:
            st.markdown("**Numerical summary**")
            try:
                numeric_view = filtered[numeric_cols].describe().T
                st.dataframe(numeric_view.round(2))
            except Exception:
                st.info("Summary statistics are not available for this dataset.")
        with right:
            st.markdown("**Detected columns**")
            if detected:
                detection_table = pd.DataFrame(
                    [{"Standard name": k, "Column in your file": v} for k, v in detected.items()])
                st.dataframe(detection_table, hide_index=True)
            else:
                st.info("No standard column names were recognised — all numeric columns were used.")

        st.markdown("**Features used for clustering**")
        st.write(f"Numerical: {', '.join(numeric_cols) if numeric_cols else 'none'}")
        st.write(f"Categorical: {', '.join(categorical_cols) if categorical_cols else 'none'}")

    # ---------------- Tab 2: cluster selection ----------------
    with tabs[1]:
        st.subheader("Choosing the Number of Clusters")
        st.markdown(
            "The **Elbow Method** looks for the point where adding another cluster stops "
            "reducing the within-cluster sum of squares significantly. The **Silhouette Score** "
            "measures how well separated the clusters are (higher is better)."
        )
        left, right = st.columns(2)
        with left:
            plot_elbow(ks, inertias)
        with right:
            plot_silhouette(ks, silhouettes)

        if ks:
            try:
                evaluation = pd.DataFrame({
                    "Clusters (k)": ks,
                    "Inertia (WCSS)": [round(v, 2) if np.isfinite(v) else np.nan for v in inertias],
                    "Silhouette Score": [round(v, 4) if np.isfinite(v) else np.nan for v in silhouettes],
                })
                st.dataframe(evaluation, hide_index=True)
            except Exception:
                pass
        st.success(f"Silhouette analysis recommends **k = {recommended_k}**. "
                   f"You are currently using **k = {n_clusters}**.")

    # ---------------- Tab 3: visual analysis ----------------
    with tabs[2]:
        st.subheader("Customer Segment Visualisations")
        plot_segment_distribution(filtered)
        st.divider()

        x_axis = "AnnualIncome" if "AnnualIncome" in filtered.columns else (
            numeric_cols[0] if numeric_cols else None)
        y_axis = "SpendingScore" if "SpendingScore" in filtered.columns else (
            numeric_cols[1] if len(numeric_cols) > 1 else None)
        if x_axis and y_axis:
            plot_scatter(filtered, x_axis, y_axis)
        else:
            st.info("Not enough numeric columns for the scatter plot.")
        st.divider()

        left, right = st.columns(2)
        with left:
            age_col = "Age" if "Age" in filtered.columns else None
            if age_col:
                plot_distribution_by_segment(filtered, age_col, kind="hist")
            else:
                st.info("No age column available for the age distribution chart.")
        with right:
            frequency_col = "PurchaseFrequency" if "PurchaseFrequency" in filtered.columns else None
            if frequency_col:
                plot_distribution_by_segment(filtered, frequency_col, kind="box")
            else:
                st.info("No purchase frequency column available.")
        st.divider()

        value_col = "AveragePurchaseValue" if "AveragePurchaseValue" in filtered.columns else None
        if value_col:
            plot_mean_bar(filtered, value_col, "Average Purchase Value by Segment")
        elif numeric_cols:
            plot_mean_bar(filtered, numeric_cols[0], f"Average {numeric_cols[0]} by Segment")
        st.divider()

        plot_segment_comparison(filtered, numeric_cols)
        st.divider()

        plot_correlation_heatmap(filtered, numeric_cols)
        st.divider()

        st.markdown("**2D PCA projection** — all scaled features compressed into two dimensions.")
        plot_pca(matrix, segmented["Cluster"].to_numpy(), segmented["Segment"].tolist())

        if "ProductCategory" in filtered.columns:
            st.divider()
            plot_category_by_segment(filtered, "ProductCategory")

    # ---------------- Tab 4: segment analysis ----------------
    with tabs[3]:
        st.subheader("Segment Statistics")
        summary = build_segment_summary(filtered)
        if summary.empty:
            st.info("Segment statistics are not available for the current selection.")
        else:
            try:
                st.dataframe(summary, hide_index=True, width="stretch")
            except Exception:
                st.dataframe(summary)

        st.divider()
        st.subheader("Automatically Generated Segment Explanations")
        segments = sorted(filtered["Segment"].astype(str).unique().tolist()) \
            if "Segment" in filtered.columns else []
        if not segments:
            st.info("No segments available to explain.")
        for segment in segments:
            insight = build_segment_insight(filtered, segment)
            with st.expander(f"🏷️ {segment} — {insight['size']} customers "
                             f"({insight['share']:.1f}%)", expanded=False):
                st.markdown(f"**Who they are:** {insight['characteristics']}")
                st.markdown(f"**How they buy:** {insight['behaviour']}")
                subset = filtered[filtered["Segment"].astype(str) == segment]
                stats_rows = []
                for column, label in STAT_LABELS:
                    if column in subset.columns and pd.api.types.is_numeric_dtype(subset[column]):
                        values = pd.to_numeric(subset[column], errors="coerce").dropna()
                        if values.empty:
                            continue
                        aggregate = values.sum() if column == "TotalPurchases" else values.mean()
                        stats_rows.append({"Metric": label, "Value": round(float(aggregate), 2)})
                if stats_rows:
                    st.dataframe(pd.DataFrame(stats_rows), hide_index=True)

    # ---------------- Tab 5: targeted insights ----------------
    with tabs[4]:
        st.subheader("🎯 Targeted Customer Insights")
        st.markdown(
            "Each recommendation below is generated from the calculated statistics of that "
            "segment relative to the rest of the customer base."
        )
        segments = sorted(filtered["Segment"].astype(str).unique().tolist()) \
            if "Segment" in filtered.columns else []
        if not segments:
            st.info("No segments available for insights.")
        for segment in segments:
            insight = build_segment_insight(filtered, segment)
            st.markdown(f"### {segment}")
            columns = st.columns(2)
            with columns[0]:
                st.markdown(f"**Customer characteristics**  \n{insight['characteristics']}")
                st.markdown(f"**Purchase behaviour**  \n{insight['behaviour']}")
                st.markdown(f"**Recommended marketing approach**  \n{insight['marketing']}")
            with columns[1]:
                st.markdown(f"**Suitable offers**  \n{insight['offers']}")
                st.markdown(f"**Business opportunity**  \n{insight['opportunity']}")
                st.metric("Share of customer base", f"{insight['share']:.1f}%")
            st.divider()

    # ---------------- Tab 6: download and documentation ----------------
    with tabs[5]:
        st.subheader("Segmented Customer Dataset")
        try:
            st.dataframe(segmented, width="stretch")
        except Exception:
            try:
                st.dataframe(segmented)
            except Exception as exc:
                st.warning(f"The dataset could not be displayed ({exc}).")

        try:
            csv_bytes = segmented.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download customer_segments.csv",
                data=csv_bytes,
                file_name="customer_segments.csv",
                mime="text/csv",
                help="Original customer information plus the cluster number and segment name.",
            )
        except Exception as exc:
            st.warning(f"The download file could not be prepared ({exc}).")

        if name_mapping:
            st.markdown("**Cluster number → segment name mapping (derived from the data)**")
            mapping_table = pd.DataFrame(
                [{"Cluster": int(k), "Segment Name": v} for k, v in sorted(name_mapping.items())])
            st.dataframe(mapping_table, hide_index=True)

        st.divider()
        render_about_section()

    st.sidebar.divider()
    st.sidebar.caption(
        "Customer Segmentation Analytics Dashboard · Built with Streamlit, pandas, "
        "NumPy, scikit-learn, matplotlib and seaborn."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # Final safety net — never show a raw traceback.
        try:
            st.error(
                "An unexpected problem occurred while running the dashboard: "
                f"{exc}. Please reload the page or try a different CSV file."
            )
        except Exception:
            print(f"Application error: {exc}")
