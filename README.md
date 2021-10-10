## How to Run
```
docker-compose build
docker-compose up
```
## How to see the results
Use a client like insomnia or Postman and send post requests like this to web component
```
curl --request POST \
  --url http://172.21.0.6:5002/prediction/results \
  --header 'Content-Type: application/json' \
  --data '{"Pclass": [3], "Sex": [0], "Age": [20], "SibSp": [0], "Parch": [0], "Embarked": [1], "Family_Size": [5], "Fare": [333], "Cabin": [5]}'
```
Make sure to replace `172.17.0.2` with the correct address


Docker swarm:
Create a docker swarm node and deploy:
```
docker swarm init
docker stack deploy --compose-file docker-compose.yaml assignment
```

Check containers with
```
docker stack services assignment
```

to view the results in swarm mode:
```
curl --request POST \
    --url http://localhost:5002/prediction/results \
    --header 'Content-Type: application/json' \
    --data '{"Pclass": [3], "Sex": [0], "Age": [20], "SibSp": [0], "Parch": [0], "Embarked": [1], "Family_Size": [5], "Fare": [333], "Cabin": [5]}'
```
