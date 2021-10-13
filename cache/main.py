import redis
from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True
redis = redis.Redis(host='redis', port=6379)


@app.route('/cache/get', methods=['POST'])
def get_cache():
    # receive the prediction request data as the message body
    hash = request.get_json()["key"]
    if redis.exists(hash):
        return redis.get(hash).decode()
    return "Not found"


@app.route('/cache/set', methods=['POST'])
def set_cache():
    # receive the prediction request data as the message body
    hash = request.get_json()["key"]
    value = request.get_json()["value"]
    if redis.exists(hash):
        return redis.get(hash).decode()
    redis.set(hash, value)
    return "OK"


app.run(host='0.0.0.0', port=5003)
