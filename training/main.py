import re

import pandas as pd
from google.cloud import storage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def fetch_train_data():
    bucket_name = "raw-titanic-data"
    source_file = "train_data.csv"
    train_data_path = "train_data.csv"
    download_blob(bucket_name, source_file, train_data_path)
    df = pd.read_csv(train_data_path)
    return df


def train_model_from_data(train_data_df):
    train_data_df.drop("Cabin", axis=1, inplace=True)

    train_data_df['Salutation'] = train_data_df['Name'].apply(split_it)

    train_data_df["Age"].fillna(
        train_data_df.groupby("Salutation")["Age"].transform("median"),
        inplace=True
    )

    train_data_df['Surname'] = train_data_df.Name.map(
        lambda x: x.split(',')[0]
    )

    to_map = {'Icard': 'C', 'Stone': 'S'}

    train_data_df.Embarked.fillna(
        train_data_df.Surname.map(to_map), inplace=True
    )

    train_data_df['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3}, inplace=True)

    train_data_df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

    train_data_df.drop(
        ['Ticket', 'PassengerId', 'Name', 'Salutation', "Fare", 'Surname'],
        axis=1,
        inplace=True
    )

    X = train_data_df.drop(['Survived'], axis='columns')
    y = train_data_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model1 = LogisticRegression()
    model1.fit(X_train, y_train)
    pred = model1.predict(X_test)
    log_acc = accuracy_score(pred, y_test)
    print(log_acc)
    return model1


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


def split_it(data):
    result = re.search('^.*,(.*)\.\s.*$', data)
    if result.group(1) not in [' Mr', ' Miss', ' Mrs', ' Master']:
        return ' Misc'
    else:
        return result.group(1)


if __name__ == "__main__":
    df = fetch_train_data()
    trained_model = train_model_from_data(df)