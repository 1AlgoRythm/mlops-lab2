import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

# The columns the model uses. Every other column in file.csv is ignored.
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

# K-Means is tried for k = 1 .. K_MAX. The elbow of that curve picks the final k.
K_MAX = 49
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

# Folder that holds the saved model and the saved scaler.
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
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
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
    raw_row = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))[FEATURES]
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
