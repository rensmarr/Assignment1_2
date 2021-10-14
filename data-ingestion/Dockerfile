FROM python:3.9-slim-buster

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install -r requirements.txt

ENV GOOGLE_APPLICATION_CREDENTIALS="/Assignment1_2/data-ingestion/key.json"

CMD [ "python", "main.py" ]
