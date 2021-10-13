## How to build and run
```
cd training
docker build . -t api
docker run api
```

## How to use
```
curl --request POST \
  --url http://172.17.0.2:5000/prediction/results \
  --header 'Content-Type: application/json' \
  --data '{"Pclass": [3], "Sex": [0], "Age": [20], "SibSp": [0], "Parch": [0], "Embarked": [1], "Family_Size": [5], "Fare": [333], "Cabin": [5]}'
```
