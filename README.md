# Airflow Lab 1 – Credit-Card Customer Clustering

[![Tests](https://github.com/1AlgoRythm/mlops-lab2/actions/workflows/tests.yml/badge.svg)](https://github.com/1AlgoRythm/mlops-lab2/actions/workflows/tests.yml)

An Apache Airflow 3 pipeline that groups credit-card customers with K-Means. It loads a CSV,
checks it, scales three columns, picks the number of clusters with the elbow method, saves the
model and predicts the cluster of one test customer. It runs in Docker once a day, and 26
tests check the code.

Based on `Labs/Airflow_Labs/Lab_1` from https://github.com/raminmohammadi/MLOps
(MLOps, DADS 7305, Northeastern University).

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Continuous Integration](#continuous-integration)
- [Testing](#testing)
- [Changes From the Upstream Lab](#changes-from-the-upstream-lab)
- [Known Limitations](#known-limitations)
- [Lessons Learned](#lessons-learned)
- [Attribution and AI Use](#attribution-and-ai-use)

## Overview

The project has four parts:

1. **`dags/airflow.py`**: the DAG (Airflow's name for a workflow). It has six tasks, a daily
   schedule and retries.
2. **`dags/src/lab.py`**: the Python function behind each task.
3. **`tests/`**: 26 tests that run without Airflow or Docker.
4. **`docker-compose.yaml`** and **`.github/workflows/`**: the Airflow setup, and the GitHub
   Actions workflow that runs the tests.

The six tasks run in this order:

1. `load_data_task`: reads `dags/data/file.csv` (8,950 customers, 18 columns).
2. `validate_data_task`: stops the run if the data is bad.
3. `data_preprocessing_task`: scales 3 columns to 0 to 1 and saves the scaler.
4. `build_save_model_task`: fits K-Means, picks k with the elbow method and saves the model.
5. `load_model_task`: predicts the cluster of one test customer.
6. `compare_models_task`: compares K-Means with a Gaussian mixture.

The model uses three columns: `BALANCE`, `PURCHASES` and `CREDIT_LIMIT`.

## Requirements

- Docker with Compose 2.14.0 or newer, and at least 4 GB of memory for Docker
- Tested with Docker 29.8.0 and Docker Compose v5.5.1
- Python 3.12 or 3.13 to run the tests outside Docker

The Airflow image is `apache/airflow:3.3.2`. The DAG needs Airflow 3.

## Installation

```bash
git clone https://github.com/1AlgoRythm/mlops-lab2.git
cd mlops-lab2
mkdir -p ./logs ./plugins ./config
docker compose up airflow-init
docker compose up -d
```

Wait for `airflow-init exited with code 0` before the last command. Then open
http://localhost:8080 and log in with username `airflow` and password `airflow`. The repository
includes a one-line `.env` (`AIRFLOW_UID=50000`), which the compose file needs.

## Usage

```bash
docker compose exec airflow-scheduler airflow dags unpause Airflow_Lab1   # a new DAG starts paused
docker compose exec airflow-scheduler airflow dags trigger Airflow_Lab1   # start a run now
docker compose down                                                       # stop Airflow
```

Unpausing starts one run at once. After that the DAG runs every day at 06:00 UTC. A run takes
about 35 seconds. Open the log of `load_model_task` to see the result:

```text
Optimal no. of clusters: 8
Saved model has 8 clusters
Scaled test row (inside the training range is 0 to 1): [0.1682, 0.0, 0.2321]
```

A run writes the model, the scaler and two JSON files (the metrics and the model comparison)
into `dags/model/`. Git ignores them.

## Project Structure

```text
mlops-lab2/
├── .github/workflows/
│   └── tests.yml              # "Tests": runs pytest on Python 3.12 and 3.13
├── dags/
│   ├── airflow.py             # the DAG: 6 tasks, schedule, retries
│   ├── data/
│   │   ├── file.csv           # 8,950 customers, 18 columns
│   │   └── test.csv           # one customer to predict
│   ├── model/                 # created by a run, not committed
│   └── src/
│       ├── __init__.py
│       └── lab.py             # the function behind each task
├── tests/
│   └── test_lab.py            # 26 tests
├── .env                       # AIRFLOW_UID=50000, needed by the compose file
├── .gitignore
├── docker-compose.yaml        # Apache's Airflow 3.3.2 file, 2 lines changed
├── pytest.ini
├── requirements.txt           # packages for the tests
└── README.md
```

## API Reference

The main functions live in `dags/src/lab.py`. Run the examples from the `dags/` folder.

### `load_data()`

Read `data/file.csv` and return it as base64 text.

### `validate_data(data_b64)`

Raise `ValueError` if a model column is missing, not numeric or negative, or if fewer than 100
rows are usable. Return a summary of the rows.

### `data_preprocessing(data_b64, scaler_filename="scaler.sav")`

Drop rows that miss a model column, scale the 3 columns to 0 to 1, save the scaler and return
the scaled data.

### `select_k(sse)`

Return the elbow k for a list of SSE values, where `sse[0]` is k = 1. Raise `ValueError` if
there is no elbow.

```python
>>> from src import lab
>>> lab.select_k([100, 50, 25, 12.5, 12, 11.8, 11.6, 11.5, 11.4, 11.3])
4
>>> lab.select_k([5.0] * 10)
Traceback (most recent call last):
    ...
ValueError: No elbow found in the SSE curve
```

### `build_save_model(data_b64, filename)`

Fit K-Means for k = 1 to 49, refit at the elbow k, save the model, write the metrics file and
return the SSE list.

### `find_out_of_range(scaler, rows)`

Return one tuple `(row index, column, value, low, high)` for every value outside the range the
scaler saw in training. An empty list means every value is inside.

```python
>>> import pandas as pd
>>> from sklearn.preprocessing import MinMaxScaler
>>> train = pd.DataFrame({"BALANCE": [0.0, 19043.0], "PURCHASES": [0.0, 49039.0], "CREDIT_LIMIT": [50.0, 30000.0]})
>>> scaler = MinMaxScaler().fit(train)
>>> outside = pd.DataFrame({"BALANCE": [3202.47], "PURCHASES": [124.5], "CREDIT_LIMIT": [241514.0]})
>>> lab.find_out_of_range(scaler, outside)
[(0, 'CREDIT_LIMIT', 241514.0, 50.0, 30000.0)]
```

### `load_model_elbow(filename, sse, scaler_filename="scaler.sav")`

Scale the customer in `data/test.csv` with the saved scaler and return its cluster as an `int`.
Raise `ValueError` if the saved model is not the elbow model, or if the customer is outside
the training range.

### `compare_models(data_b64, sse)`

Fit a Gaussian mixture with the elbow k and compare it with K-Means by silhouette score. Write
a comparison file and return the same values. K-Means stays the model that is used.

## Continuous Integration

### How it works

1. A push to `main`, or a pull request, triggers the workflow.
2. GitHub starts a fresh Ubuntu machine for each Python version in the matrix.
3. The machine checks out the code and installs the requested Python version.
4. It installs the packages in `requirements.txt`.
5. It runs the tests and writes an XML report.
6. GitHub shows a green check if every step passed and a red cross if any step failed.

### Workflows

| Workflow | File | Runs on | Command | Artifacts |
|---|---|---|---|---|
| Tests | `tests.yml` | push to `main`; pull requests | `pytest --junitxml=pytest-report.xml` | `test-results-3.12`, `test-results-3.13` |

### Python version matrix

The job is written once, and GitHub runs it once per version in `strategy.matrix.python-version`:
**3.12** and **3.13**. Python 3.13 is the version inside the Airflow image. Versions are written
in quotes, and `fail-fast: false` lets one version fail without cancelling the other.

The XML report of each version is uploaded as `test-results-<version>`. To download one, open a
run on the **Actions** tab and scroll to **Artifacts**.

## Testing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

| File | Tests | Original lab |
|---|---|---|
| `tests/test_lab.py` | 26 | 0 |

The tests cover three kinds of cases:

- **Normal cases:** a good table passes the check, the scaler is saved, the saved model has the
  elbow number of clusters, and the metrics and comparison files are written.
- **Error cases:** a missing column, a text column, too few rows, negative values, a curve
  without an elbow, a saved model with the wrong number of clusters, and a test customer outside
  the training range.
- **Scaling:** the test customer is scaled with the training scaler before the model predicts.
  A scaled maximum came out as 1.0000000000000002, so the test allows a tolerance.

To check that the tests can fail, they were run against six deliberately broken copies of
`lab.py`. All six are caught. The suite passes on Python 3.12 and on Python 3.13 (inside the
Airflow image).

## Changes From the Upstream Lab

Each change is its own commit.

### Fixed

- **The saved model was the wrong model.** The lab saved the last model tried, k = 49, while its
  log said the best k was 8. It now saves the model refitted at the elbow k
  ([`255a295`](https://github.com/1AlgoRythm/mlops-lab2/commit/255a295)).
- **The test customer was not scaled.** The model learned from data scaled to 0 to 1, but the lab
  predicted on the raw row. The scaler is now saved and used
  ([`c3ab987`](https://github.com/1AlgoRythm/mlops-lab2/commit/c3ab987)).
- **The test customer was outside the data.** Its credit limit, 241514, is far above the largest
  in the data (30000). `test.csv` now holds customer C10002's real values
  ([`22963f5`](https://github.com/1AlgoRythm/mlops-lab2/commit/22963f5)).
- **Rows were dropped for a column the model never reads.** `dropna()` removed 314 rows; it now
  looks at the 3 model columns and removes 1
  ([`5947650`](https://github.com/1AlgoRythm/mlops-lab2/commit/5947650)).

### Added

- **An input check:** the new `validate_data_task`, and a range check on the test customer
  ([`eab5054`](https://github.com/1AlgoRythm/mlops-lab2/commit/eab5054)).
- **A metrics file for every run**, with k, inertia and silhouette score
  ([`c5b74e0`](https://github.com/1AlgoRythm/mlops-lab2/commit/c5b74e0)).
- **A second model for comparison:** a Gaussian mixture, compared with K-Means by silhouette
  score. On this data K-Means scores 0.4015 and the mixture scores -0.0179, so K-Means stays
  the saved model
  ([`89ff84e`](https://github.com/1AlgoRythm/mlops-lab2/commit/89ff84e)).
- **Tests:** 22 tests and a `requirements.txt`, and 4 more with the model comparison
  ([`6d37f86`](https://github.com/1AlgoRythm/mlops-lab2/commit/6d37f86)).
- **A daily schedule, retries and a time limit:** daily at 06:00 UTC, 2 retries one minute apart,
  and 10 minutes per task
  ([`b93d086`](https://github.com/1AlgoRythm/mlops-lab2/commit/b93d086)).
- **A GitHub Actions workflow** that runs the tests on Python 3.12 and 3.13
  ([`36ade5f`](https://github.com/1AlgoRythm/mlops-lab2/commit/36ade5f)).

### Changed

- **Owner and description** of the DAG
  ([`5aa8c9a`](https://github.com/1AlgoRythm/mlops-lab2/commit/5aa8c9a)).
- **The compose file** differs from Apache's 3.3.2 file by two lines: the example DAGs are off
  ([`5988d69`](https://github.com/1AlgoRythm/mlops-lab2/commit/5988d69)), and the packages the
  DAG needs install by default
  ([`f58e836`](https://github.com/1AlgoRythm/mlops-lab2/commit/f58e836)).
- **A one-line `.env`.** Apache's compose file stops with `env file .env not found` without
  one. This showed up when the README steps were run on a fresh clone
  ([`1442d00`](https://github.com/1AlgoRythm/mlops-lab2/commit/1442d00)).

## Known Limitations

- **The DAG file is called `airflow.py`**, the same name as the Airflow package. It works inside
  Airflow, but it hides the package if the `dags/` folder comes before Airflow on Python's path.
- **The packages are not pinned.** The containers install the newest `pandas`, `scikit-learn`
  and `kneed` at every start. The tested versions were pandas 3.0.5, scikit-learn 1.9.1 and
  kneed 0.8.6.
- **The login `airflow` / `airflow` is a demo login.** The Airflow documentation says this
  compose setup is not for production.
- **The test customer is part of the training data.** It checks that the pipeline works. It does
  not measure how well the model predicts.
- **The model comparison is advisory.** The Gaussian mixture never replaces K-Means.
- **GitHub Actions has not been run on this code at the time of writing.** The workflow's
  commands were run in a fresh environment and passed.

## Lessons Learned

- A log line is not proof. The lab printed "optimal 8" while the saved model had 49 clusters.
- A model must see new data the way it saw training data. Save the scaler with the model.
- A pipeline should refuse bad input with a clear message, not answer with a guess.
- A test that passes is only useful if it can fail.
- Run the README on a fresh clone. It showed that the compose file needs a `.env`.

## Attribution and AI Use

This repository is based on the lab in the course repository linked at the top. Claude (an
AI assistant) was used to draft the workflow. The code was run, and the results were checked
in the Airflow web pages and in the logs.

---

# Original lab README (kept for reference)

# Airflow lab

- In order to install Airflow using docker you can watch our [Airflow Lab1 Tutorial Video](https://youtu.be/exFSeGUbn4Q?feature=shared)
- For latest step-by-step instructions, check out this blog - [AirFlow Lab-1](https://www.mlwithramin.com/blog/airflow-lab1)

### ML Model

This script is designed for data clustering using K-Means clustering and determining the optimal number of clusters using the elbow method. It provides functionality to load data from a CSV file, perform data preprocessing, build and save a K-Means clustering model, and determine the number of clusters based on the elbow method.

#### Prerequisites

Before using this script, make sure you have the following libraries installed:

- pandas
- scikit-learn (sklearn)
- kneed
- pickle

#### Usage

You can use this script to perform K-Means clustering on your dataset as follows:

```python
# Load the data
data = load_data()

# Preprocess the data
preprocessed_data = data_preprocessing(data)

# Build and save the clustering model
sse_values = build_save_model(preprocessed_data, 'clustering_model.pkl')

# Load the saved model and determine the number of clusters
result = load_model_elbow('clustering_model.pkl', sse_values)
print(result)
```

#### Functions

1. **load_data():**
   - *Description:* Loads data from a CSV file, serializes it, and returns the serialized data.
   - *Usage:*
     ```python
     data = load_data()
     ```

2. **data_preprocessing(data)**
   - *Description:* Deserializes data, performs data preprocessing, and returns serialized clustered data.
   - *Usage:*
     ```python
     preprocessed_data = data_preprocessing(data)
     ```

3. **build_save_model(data, filename)**
   - *Description:* Builds a K-Means clustering model, saves it to a file, and returns SSE values.
   - *Usage:*
     ```python
     sse_values = build_save_model(preprocessed_data, 'clustering_model.pkl')
     ```

4. **load_model_elbow(filename, sse)**
   - *Description:* Loads a saved K-Means clustering model and determines the number of clusters using the elbow method.
   - *Usage:*
     ```python
     result = load_model_elbow('clustering_model.pkl', sse_values)
     ```
### Airflow Setup

Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

References

-   Product - https://airflow.apache.org/
-   Documentation - https://airflow.apache.org/docs/
-   Github - https://github.com/apache/airflow

#### Installation

Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).

Local

-   Docker Desktop Running

Cloud

-   Linux VM
-   SSH Connection
-   Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

#### Tutorial

1. Create a new directory

    ```bash
    mkdir -p ~/app
    cd ~/app
    ```

2. Running Airflow in Docker - [Refer](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-airflow-in-docker)

    a. You can check if you have enough memory by running this command

    ```bash
    docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
    ```

    b. Fetch [docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml)

    ```bash
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml'
    ```

    c. Setting the right Airflow user

    ```bash
    mkdir -p ./dags ./logs ./plugins ./working_data
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```

    d. Update the following in docker-compose.yml

    ```bash
    # Donot load examples
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

    # Additional python package
    _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas }

    # Output dir
    - ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

    # Change default admin credentials
    _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
    _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
    ```

    e. Initialize the database

    ```bash
    docker compose up airflow-init
    ```

    f. Running Airflow

    ```bash
    docker compose up
    ```

    Wait until terminal outputs

    `app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

    g. Enable port forwarding

    h. Visit `localhost:8080` login with credentials set on step `2.d`

3. Explore UI and add user `Security > List Users`

4. Create a python script [`dags/sandbox.py`](dags/sandbox.py)

    - BashOperator
    - PythonOperator
    - Task Dependencies
    - Params
    - Crontab schedules

    You can have n number of scripts inside dags dir

5. Stop docker containers

    ```bash
    docker compose down
    ```
### Airflow DAG Script

This Markdown file provides a detailed explanation of the Python script that defines an Airflow Directed Acyclic Graph (DAG) for a data processing and modeling workflow.

#### Script Overview

The script defines an Airflow DAG named `your_python_dag` that consists of several tasks. Each task represents a specific operation in a data processing and modeling workflow. The script imports necessary libraries, sets default arguments for the DAG, creates PythonOperators for each task, defines task dependencies, and provides command-line interaction with the DAG.

#### Importing Libraries

```python
# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf
```
The script starts by importing the required libraries and modules. Notable imports include the `DAG` and `PythonOperator` classes from the `airflow` package, datetime manipulation functions, and custom functions from the `src.lab` module.



#### Enable pickle support for XCom, allowing data to be passed between tasks
```python
conf.set('core', 'enable_xcom_pickling', 'True')
```

#### Define default arguments for your DAG
```python
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}
```
Default arguments for the DAG are specified in a dictionary named default_args. These arguments include the DAG owner's name, the start date, the number of retries, and the retry delay in case of task failure.

#### Create a DAG instance named 'your_python_dag' with the defined default arguments
``` python 
dag = DAG(
    'your_python_dag',
    default_args=default_args,
    description='Your Python DAG Description',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)
```
Here, the DAG object dag is created with the name 'your_python_dag' and the specified default arguments. The description provides a brief description of the DAG, and schedule_interval defines the execution schedule (in this case, it's set to None for manual triggering). catchup is set to False to prevent backfilling of missed runs.


#### Task to load data, calls the 'load_data' Python function
``` python 
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)
```

#### Task to perform data preprocessing, depends on 'load_data_task'
```python 
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)
```
The 'data_preprocessing_task' depends on the 'load_data_task' and calls the data_preprocessing function, which is provided with the output of the 'load_data_task'.

#### Task to build and save a model, depends on 'data_preprocessing_task'
```python
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)
```
The 'build_save_model_task' depends on the 'data_preprocessing_task' and calls the build_save_model function. It also provides additional context information and arguments.

#### Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
``` python
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)
```
The 'load_model_task' depends on the 'build_save_model_task' and calls the load_model_elbow function with specific arguments.

#### Set task dependencies
```python
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
```
Task dependencies are defined using the >> operator. In this case, the tasks are executed in sequence: 'load_data_task' -> 'data_preprocessing_task' -> 'build_save_model_task' -> 'load_model_task'.

#### If this script is run directly, allow command-line interaction with the DAG
```python
if __name__ == "__main__":
    dag.cli()
```
- Lastly, the script allows for command-line interaction with the DAG. When the script is run directly, the dag.cli() function is called, providing the ability to trigger and manage the DAG from the command line.
- This script defines a comprehensive Airflow DAG for a data processing and modeling workflow, with clear task dependencies and default arguments.

### Running an Apache Airflow DAG Pipeline in Docker

This guide provides detailed steps to set up and run an Apache Airflow Directed Acyclic Graph (DAG) pipeline within a Docker container using Docker Compose. The pipeline is named "your_python_dag."

#### Prerequisites

- Docker: Make sure Docker is installed and running on your system.

#### Step 1: Directory Structure

Ensure your project has the following directory structure:

```plaintext
your_airflow_project/
├── dags/
│   ├── airflow.py     # Your DAG script
├── src/
│   ├── lab.py                # Data processing and modeling functions
├── data/                       # Directory for data (if needed)
├── docker-compose.yaml         # Docker Compose configuration
```

#### Step 2: Docker Compose Configuration
Create a docker-compose.yaml file in the project root directory. This file defines the services and configurations for running Airflow in a Docker container.

#### Step 3: Start the Docker containers by running the following command

```plaintext
docker compose up
```

Wait until you see the log message indicating that the Airflow webserver is running:

```plaintext
app-airflow-webserver-1 | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"
```

#### Step 4: Access Airflow Web Interface
- Open a web browser and navigate to http://localhost:8080.

- Log in with the credentials set in the .env file or use the default credentials (username: admin, password: admin).

- Once logged in, you'll be on the Airflow web interface.

#### Step 5: Trigger the DAG
- In the Airflow web interface, navigate to the "DAGs" page.

- You should see the "your_python_dag" listed.

- To manually trigger the DAG, click on the "Trigger DAG" button or enable the DAG by toggling the switch to the "On" position.

- Monitor the progress of the DAG in the Airflow web interface. You can view logs, task status, and task execution details.

#### Step 6: Pipeline Outputs

- Once the DAG completes its execution, check any output or artifacts produced by your functions and tasks. 