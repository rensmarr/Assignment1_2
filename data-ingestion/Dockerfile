FROM python:3.9-slim-buster

WORKDIR /usr/src/app

COPY . /usr/src/app
COPY . /usr/src/app/ 

RUN pip install -r requirements.txt

ENV GOOGLE_APPLICATION_CREDENTIALS="/usr/src/app/key.json"

CMD [ "python", "main.py" ]
