import pandas as pd
from google.cloud import storage
import numpy as np


def fetch_train_data():
    bucket_name = "raw-titanic-data"
    source_file = "train_data.csv"
    train_data_path = "train_data.csv"
    download_blob(bucket_name, source_file, train_data_path)
    df = pd.read_csv(train_data_path)
    return df


def feature_data(dataframe):
    extract_title(df)
    extract_deck(df)
    extract_family(df)
    extract_age_class(df)
    extract_fare_per_person(df)
    print(df)

    return dataframe


def extract_title(df: pd.DataFrame):
    title_list = [
        'Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle',
        'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer'
    ]
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title'] = df.apply(replace_titles, axis=1)


def substrings_in_string(big_string: str, substrings):
    if type(big_string) is float:
        return np.nan
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


def extract_deck(df):
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))


def extract_family(df):
    df['Family_Size'] = df['SibSp'] + df['Parch']


def extract_age_class(df):
    df['Age_Class'] = df['Age'] * df['Pclass']


def extract_fare_per_person(df):
    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)


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


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


if __name__ == "__main__":
    df = fetch_train_data()
    df = feature_data(df)
    featured_data_path = "titanic_featured_data.csv"
    df.to_csv(featured_data_path)
    upload_blob(
        "feature-titanic-data", featured_data_path, "titanic_featured_data.csv"
    )
