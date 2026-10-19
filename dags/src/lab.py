import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import base64
import json
from datetime import datetime, timezone

# The columns the model uses. Every other column in file.csv is ignored.
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

# K-Means is tried for k = 1 .. K_MAX. The elbow of that curve picks the final k.
K_MAX = 49
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

# Folders next to this package: the input CSV files, and the saved model and scaler.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")


def select_k(sse):
    """
    Returns the elbow k for a list of SSE values, where sse[0] is k = 1.
    Raises ValueError if the curve has no elbow, so a bad curve cannot pass silently.
    """
    knee = KneeLocator(range(1, len(sse) + 1), sse, curve="convex", direction="decreasing")
    if knee.elbow is None:
        raise ValueError("No elbow found in the SSE curve")
    return int(knee.elbow)


# A clustering run on fewer usable rows than this is not meaningful.
MIN_ROWS = 100


def validate_data(data_b64: str):
    """
    Checks the loaded table before the model sees it. Raises ValueError with a clear
    message if a model column is missing or not numeric, if there are too few usable
    rows, or if a model column has negative values.
    Returns a small JSON-safe summary.
    """
    df = pickle.loads(base64.b64decode(data_b64))

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    not_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        raise ValueError(f"Columns that are not numeric: {not_numeric}")

    usable = df.dropna(subset=FEATURES)
    if len(usable) < MIN_ROWS:
        raise ValueError(f"Only {len(usable)} usable rows; at least {MIN_ROWS} are needed")

    negative = [c for c in FEATURES if (usable[c] < 0).any()]
    if negative:
        raise ValueError(f"Negative values in: {negative}")

    summary = {
        "rows": int(len(df)),
        "usable_rows": int(len(usable)),
        "ranges": {c: [float(usable[c].min()), float(usable[c].max())] for c in FEATURES},
    }
    print(f"Data check passed: {summary}")
    return summary


def find_out_of_range(scaler, rows):
    """
    Returns (row index, column, value, low, high) for every value in rows that lies
    outside the range the scaler saw in training. An empty list means every value is inside.
    """
    problems = []
    for j, column in enumerate(FEATURES):
        low, high = float(scaler.data_min_[j]), float(scaler.data_max_[j])
        for i, value in enumerate(rows[column]):
            if value < low or value > high:
                problems.append((i, column, float(value), low, high))
    return problems


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(DATA_DIR, "file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str, scaler_filename: str = "scaler.sav"):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    The fitted scaler is saved next to the model, so new rows can be scaled the same way.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # Drop a row only if one of the columns the model uses is missing.
    rows_before = len(df)
    df = df.dropna(subset=FEATURES)
    print(f"Rows before: {rows_before}, after dropping missing values: {len(df)}")
    clustering_data = df[FEATURES]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, scaler_filename), "wb") as f:
        pickle.dump(min_max_scaler, f)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def write_metrics(scaled_data, model, best_k):
    """
    Writes a metrics file for this run, named with the run's UTC time, and returns the metrics.
    The silhouette score runs from -1 to 1; higher means the clusters are better separated.
    It needs at least 2 clusters, so it is None when the elbow is k = 1.
    """
    now = datetime.now(timezone.utc)
    metrics = {
        "run_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(scaled_data)),
        "k": int(best_k),
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(scaled_data, model.labels_)) if best_k >= 2 else None,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"metrics_{now.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {os.path.basename(path)}: {metrics}")
    return metrics


def build_save_model(data_b64: str, filename: str):
    """
    Tries k = 1 .. K_MAX, picks the elbow k, refits K-Means at that k and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    sse = []
    for k in range(1, K_MAX + 1):
        sse.append(KMeans(n_clusters=k, **KMEANS_KWARGS).fit(df).inertia_)

    # The original code saved the last model tried (k=49), not the elbow model.
    best_k = select_k(sse)
    model = KMeans(n_clusters=best_k, **KMEANS_KWARGS).fit(df)
    print(f"Elbow k = {best_k}; saved model has {model.n_clusters} clusters")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(model, f)

    write_metrics(df, model, best_k)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list, scaler_filename: str = "scaler.sav"):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved model
    loaded_model = pickle.load(open(os.path.join(MODEL_DIR, filename), "rb"))

    # the saved model must be the elbow model; stop the run if it is not
    best_k = select_k(sse)
    print(f"Optimal no. of clusters: {best_k}")
    print(f"Saved model has {loaded_model.n_clusters} clusters")
    if loaded_model.n_clusters != best_k:
        raise ValueError(
            f"Saved model has {loaded_model.n_clusters} clusters but the elbow is {best_k}"
        )

    # The model learned from scaled data (0 to 1), so the test row must be scaled the
    # same way. The original code predicted on the raw row.
    scaler = pickle.load(open(os.path.join(MODEL_DIR, scaler_filename), "rb"))
    raw_row = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))[FEATURES]
    scaled_row = scaler.transform(raw_row)
    print(f"Scaled test row (inside the training range is 0 to 1): {scaled_row[0].round(4).tolist()}")

    # Refuse a row outside the training range: the model would still answer, but the answer
    # would be a guess far from anything it has seen.
    problems = find_out_of_range(scaler, raw_row)
    if problems:
        details = "; ".join(
            f"{column}={value} (training range {low} to {high})"
            for _, column, value, low, high in problems
        )
        raise ValueError(f"Test row is outside the training range: {details}")
    pred = loaded_model.predict(scaled_row)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred


def silhouette_or_none(scaled_data, labels):
    """
    Returns the silhouette score of a labelling, or None when it is undefined: the score
    needs at least 2 different labels and fewer labels than rows.
    """
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(scaled_data):
        return None
    return float(silhouette_score(scaled_data, labels))


def compare_models(data_b64: str, sse: list):
    """
    Fits a second model, a Gaussian mixture, with the elbow k and compares it with K-Means
    on the same scaled data. The comparison is written to model/model_comparison_TIME.json.
    It changes nothing else: K-Means stays the saved and used model.
    Returns the comparison (JSON-safe).
    """
    scaled = pickle.loads(base64.b64decode(data_b64))
    k = select_k(sse)

    kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(scaled)
    mixture = GaussianMixture(n_components=k, random_state=42).fit(scaled)
    kmeans_score = silhouette_or_none(scaled, kmeans.labels_)
    mixture_score = silhouette_or_none(scaled, mixture.predict(scaled))

    if kmeans_score is None or mixture_score is None:
        better = None
    else:
        better = "kmeans" if kmeans_score >= mixture_score else "gaussian_mixture"

    now = datetime.now(timezone.utc)
    comparison = {
        "run_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(scaled)),
        "k": int(k),
        "kmeans_silhouette": kmeans_score,
        "gaussian_mixture_silhouette": mixture_score,
        "gaussian_mixture_converged": bool(mixture.converged_),
        "better_by_silhouette": better,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"model_comparison_{now.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Model comparison written to {os.path.basename(path)}: {comparison}")
    return comparison
