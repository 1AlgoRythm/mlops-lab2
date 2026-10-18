"""Tests for dags/src/lab.py. They need no Airflow and no Docker."""
import base64
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src import lab


GROUP_TWO = (15000, 50, 1000)  # balance, purchases, credit limit at the centre of group two


def encode(obj):
    """Same encoding the DAG uses to pass data between tasks."""
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def make_table(n_per_blob=60, seed=0):
    """
    Three groups of customers, far apart in every direction (so the elbow is at k = 3),
    plus two columns the model ignores. Group two is centred at GROUP_TWO.
    """
    rng = np.random.default_rng(seed)
    centers = [(100, 50, 25000), GROUP_TWO, (7500, 30000, 13000)]
    blocks = [rng.normal(loc=c, scale=[50, 20, 200], size=(n_per_blob, 3)) for c in centers]
    values = np.clip(np.vstack(blocks), 1, None)
    df = pd.DataFrame(values, columns=lab.FEATURES)
    df["CUST_ID"] = [f"C{i:05d}" for i in range(len(df))]
    df["TENURE"] = 12
    return df


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Point the module at temporary folders and a small k range, so tests are fast."""
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(lab, "MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(lab, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(lab, "K_MAX", 8)
    return tmp_path


def fit_pipeline(df):
    """Preprocess and build, as the DAG does. Returns (scaled data as base64, sse list)."""
    scaled = lab.data_preprocessing(encode(df))
    return scaled, lab.build_save_model(scaled, "model.sav")


# ---- select_k ------------------------------------------------------------------------------

def test_select_k_finds_the_bend():
    sse = [100, 50, 25, 12.5, 12, 11.8, 11.6, 11.5, 11.4, 11.3]
    assert lab.select_k(sse) == 4


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # kneed divides by zero on a flat curve
@pytest.mark.parametrize(
    "sse", [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [5.0] * 10, [10, 1]], ids=["linear", "flat", "two-points"]
)
def test_select_k_rejects_a_curve_without_an_elbow(sse):
    with pytest.raises(ValueError, match="No elbow"):
        lab.select_k(sse)


# ---- validate_data -------------------------------------------------------------------------

def test_validate_data_accepts_a_good_table():
    df = make_table()
    summary = lab.validate_data(encode(df))
    assert summary["rows"] == len(df)
    assert summary["usable_rows"] == len(df)
    assert set(summary["ranges"]) == set(lab.FEATURES)
    json.dumps(summary)  # the summary goes through XCom, so it must be JSON-safe


def test_validate_data_rows_missing_only_an_unused_column_are_usable():
    df = make_table()
    df.loc[:4, "TENURE"] = np.nan
    assert lab.validate_data(encode(df))["usable_rows"] == len(df)


def test_validate_data_rejects_a_missing_column():
    df = make_table().drop(columns=["BALANCE"])
    with pytest.raises(ValueError, match=r"Missing columns: \['BALANCE'\]"):
        lab.validate_data(encode(df))


def test_validate_data_rejects_a_text_column():
    df = make_table()
    df["PURCHASES"] = df["PURCHASES"].astype(str)
    with pytest.raises(ValueError, match="not numeric"):
        lab.validate_data(encode(df))


def test_validate_data_rejects_too_few_rows():
    df = make_table(n_per_blob=10)  # 30 rows, below MIN_ROWS
    with pytest.raises(ValueError, match="usable rows"):
        lab.validate_data(encode(df))


def test_validate_data_rejects_negative_values():
    df = make_table()
    df.loc[0, "CREDIT_LIMIT"] = -5
    with pytest.raises(ValueError, match="Negative values"):
        lab.validate_data(encode(df))


# ---- data_preprocessing --------------------------------------------------------------------

def test_preprocessing_drops_only_rows_missing_a_model_column(workdir):
    df = make_table()
    df.loc[[0, 1], "CREDIT_LIMIT"] = np.nan  # missing a model column: dropped
    df.loc[2:6, "TENURE"] = np.nan  # missing an unused column: kept
    scaled = pickle.loads(base64.b64decode(lab.data_preprocessing(encode(df))))
    assert scaled.shape == (len(df) - 2, 3)  # (rows, features)
    # scaling gives 0 to 1; allow floating-point rounding (the max came out as 1.0000000000000002)
    assert scaled.min() >= -1e-9 and scaled.max() <= 1 + 1e-9


def test_preprocessing_saves_the_scaler_it_fitted(workdir):
    df = make_table()
    lab.data_preprocessing(encode(df))
    scaler = pickle.load(open(workdir / "model" / "scaler.sav", "rb"))
    assert scaler.data_min_.tolist() == pytest.approx(df[lab.FEATURES].min().tolist())
    assert scaler.data_max_.tolist() == pytest.approx(df[lab.FEATURES].max().tolist())


# ---- find_out_of_range ---------------------------------------------------------------------

@pytest.fixture
def lab_range_scaler():
    """A scaler fitted on the real ranges of the lab's data: credit limit 50 to 30000."""
    frame = pd.DataFrame(
        {"BALANCE": [0.0, 19043.0], "PURCHASES": [0.0, 49039.0], "CREDIT_LIMIT": [50.0, 30000.0]}
    )
    return MinMaxScaler().fit(frame[lab.FEATURES])


def test_find_out_of_range_accepts_values_inside_and_on_the_edge(lab_range_scaler):
    rows = pd.DataFrame({"BALANCE": [0.0, 19043.0], "PURCHASES": [0.0, 10.0], "CREDIT_LIMIT": [50.0, 30000.0]})
    assert lab.find_out_of_range(lab_range_scaler, rows) == []


def test_find_out_of_range_flags_the_original_lab_test_row(lab_range_scaler):
    """The lab's first test.csv row had a credit limit of 241514. Only that value is outside."""
    row = pd.DataFrame({"BALANCE": [3202.467416], "PURCHASES": [124.5], "CREDIT_LIMIT": [241514.0]})
    problems = lab.find_out_of_range(lab_range_scaler, row)
    assert problems == [(0, "CREDIT_LIMIT", 241514.0, 50.0, 30000.0)]


# ---- build_save_model ----------------------------------------------------------------------

def test_saved_model_has_the_elbow_number_of_clusters(workdir):
    _, sse = fit_pipeline(make_table())
    assert len(sse) == lab.K_MAX
    model = pickle.load(open(workdir / "model" / "model.sav", "rb"))
    assert model.n_clusters == lab.select_k(sse) == 3  # three groups were generated


def test_a_metrics_file_is_written_for_the_run(workdir):
    df = make_table()
    fit_pipeline(df)
    files = glob.glob(str(workdir / "model" / "metrics_*.json"))
    assert len(files) == 1
    metrics = json.load(open(files[0]))
    assert set(metrics) == {"run_utc", "rows", "k", "inertia", "silhouette"}
    assert metrics["rows"] == len(df) and metrics["k"] == 3
    assert 0 < metrics["silhouette"] <= 1  # well separated groups score high


def test_silhouette_is_none_when_there_is_only_one_cluster(workdir):
    scaled = MinMaxScaler().fit_transform(make_table()[lab.FEATURES])
    metrics = lab.write_metrics(scaled, KMeans(n_clusters=1, n_init=1).fit(scaled), 1)
    assert metrics["silhouette"] is None


# ---- load_model_elbow ----------------------------------------------------------------------

def write_test_row(workdir, balance, purchases, credit_limit):
    path = workdir / "data" / "test.csv"
    path.write_text(f"BALANCE,PURCHASES,CREDIT_LIMIT\n{balance},{purchases},{credit_limit}\n")


def test_prediction_matches_the_cluster_of_the_group_it_belongs_to(workdir):
    df = make_table()
    _, sse = fit_pipeline(df)
    model = pickle.load(open(workdir / "model" / "model.sav", "rb"))
    group_two_label = set(model.labels_[60:120])  # rows 60 to 119 were generated around group two
    assert len(group_two_label) == 1
    write_test_row(workdir, *GROUP_TWO)  # the centre of group two
    assert lab.load_model_elbow("model.sav", sse) == group_two_label.pop()


def test_the_row_is_scaled_with_the_training_scaler_before_predicting(workdir, monkeypatch):
    """Checks the numbers the model receives; the right cluster alone can be luck."""
    df = make_table()
    _, sse = fit_pipeline(df)
    received = []
    real_predict = KMeans.predict

    def spy(self, X, *args, **kwargs):
        received.append(np.asarray(X))
        return real_predict(self, X, *args, **kwargs)

    monkeypatch.setattr(KMeans, "predict", spy)
    write_test_row(workdir, *GROUP_TWO)
    lab.load_model_elbow("model.sav", sse)
    expected = MinMaxScaler().fit(df[lab.FEATURES]).transform(pd.DataFrame([GROUP_TWO], columns=lab.FEATURES))
    assert received[0] == pytest.approx(expected)  # shape (1, 3): one row, three scaled features
    assert received[0].max() <= 1


def test_load_model_elbow_refuses_a_row_outside_the_training_range(workdir):
    _, sse = fit_pipeline(make_table())
    write_test_row(workdir, GROUP_TWO[0], GROUP_TWO[1], 241514)
    with pytest.raises(ValueError, match=r"CREDIT_LIMIT=241514\.0"):
        lab.load_model_elbow("model.sav", sse)


def test_load_model_elbow_stops_when_the_saved_model_is_not_the_elbow_model(workdir):
    scaled, sse = fit_pipeline(make_table())
    wrong = KMeans(n_clusters=5, n_init=1, random_state=0).fit(pickle.loads(base64.b64decode(scaled)))
    pickle.dump(wrong, open(workdir / "model" / "model.sav", "wb"))
    write_test_row(workdir, *GROUP_TWO)
    with pytest.raises(ValueError, match="Saved model has 5 clusters"):
        lab.load_model_elbow("model.sav", sse)


# ---- the real lab data ---------------------------------------------------------------------

def test_the_real_data_passes_the_check_and_the_real_test_row_is_in_range():
    summary = lab.validate_data(lab.load_data())
    assert summary["rows"] == 8950 and summary["usable_rows"] == 8949
    test_row = pd.read_csv(os.path.join(lab.DATA_DIR, "test.csv"))
    for column in lab.FEATURES:
        low, high = summary["ranges"][column]
        assert low <= test_row[column].iloc[0] <= high
