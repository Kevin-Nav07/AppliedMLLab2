

# Toxic Model Service (Dockerized)



## Overview
Serves the Hugging Face model `unitary/toxic-bert` behind a REST API. with two endpoints
"/health" which checks the status of the server and "/predict" which takes in a json {"text:"sample text"} and then sends a response by outputting the toxicity label, the toxicity score as well as the threshold for a toxic score to be toxic or non-toxic.

## Requirements
- Docker Desktop (or Docker Engine)


## Pull the image
```bash 
docker pull kevinnav07/toxic-model-service:latest
```

## Run the image as a Container
```bash
docker run --rm -p 5000:5000 kevinnav07/toxic-model-service:latest
```

the server for the api should be up and receptive now in a seperate terminal

run to check the API is working

```bash
curl http://localhost:5000/health
```

and that should print an ok message

then run this prompts to measure the toxicity of these statements
but change the the text inside to get various results.
For instance, "you are disgusting and useless" should output toxic
and "hello how are you" should be non-toxic

NOTE: the toxic model taken from hugging face gives a toxicity score for the text, but I manually added a response to indicate whether that means toxic or not based off some arbitrary score which is it has to have a threshold of  >= 0.1 in order to be classified as toxic. This threshold can be changed in the app.py file

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"You are disgusting and useless.\"}"
```

docker hub link: docker hub link: https://hub.docker.com/r/kevinnav07/toxic-model-service