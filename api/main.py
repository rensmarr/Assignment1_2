import hashlib
import json
import pickle
from typing import Any, Dict

import pandas as pd
import redis
from flask import Flask, json, request
from google.cloud import storage


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def download_model():
    trained_model_pickle_path = "trained_model.pkl"
    download_blob(
        "trained-titanic-model",
        "logistic_regression.pkl",
        trained_model_pickle_path,
    )
    model = None
    with open(trained_model_pickle_path, 'rb') as f:
        model = pickle.load(f)
    return model


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


app = Flask(__name__)
app.config["DEBUG"] = True
model = download_model()
redis = redis.Redis(host='redis', port=6379)


@app.route('/prediction/results', methods=['POST'])
def predict_perf():
    # receive the prediction request data as the message body
    content = request.get_json()
    hash = dict_hash(content)
    if redis.exists(hash):
        return redis.get(hash).decode()
    df = pd.read_json(json.dumps(content), orient='records')
    resp = model.predict(df)
    redis.set(hash, str(resp))
    return str(resp)


app.run(host='0.0.0.0', port=5002)
