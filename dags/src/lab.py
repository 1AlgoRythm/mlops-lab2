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


def select_k(sse):
    """
    Returns the elbow k for a list of SSE values, where sse[0] is k = 1.
    Raises ValueError if the curve has no elbow, so a bad curve cannot pass silently.
    """
    knee = KneeLocator(range(1, len(sse) + 1), sse, curve="convex", direction="decreasing")
    if knee.elbow is None:
        raise ValueError("No elbow found in the SSE curve")
    return int(knee.elbow)


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

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
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

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # the saved model must be the elbow model; stop the run if it is not
    best_k = select_k(sse)
    print(f"Optimal no. of clusters: {best_k}")
    print(f"Saved model has {loaded_model.n_clusters} clusters")
    if loaded_model.n_clusters != best_k:
        raise ValueError(
            f"Saved model has {loaded_model.n_clusters} clusters but the elbow is {best_k}"
        )

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
