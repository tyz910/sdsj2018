IMAGE=tyz910/sdsj2018
DOWNLOAD_URL=https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip

ifeq ($(OS), Windows_NT)
	PWD=${CURDIR}
	DOCKER_BUILD=docker build -t ${IMAGE} .
else
	DOCKER_BUILD=docker build -t ${IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images
endif

ifeq ($(DATASET),)
	DATASET=1
endif

DATASET_MODE=$(shell ls -la ./data | grep "check_${DATASET}_*" | grep -o "_[rc]")
DATASET_NAME=${DATASET}${DATASET_MODE}

ifeq ($(DATASET_MODE), _r)
	TRAIN_MODE=regression
else
	TRAIN_MODE=classification
endif

TRAIN_CSV=data/check_${DATASET_NAME}/train.csv
TEST_CSV=data/check_${DATASET_NAME}/test.csv
PREDICTIONS_CSV=predictions/check_${DATASET_NAME}.csv
MODEL_DIR=models/check_${DATASET_NAME}

SUBMISSION_TIME=$(shell date '+%Y%m%d_%H%M%S')
SUBMISSION_FILE=submission_${SUBMISSION_TIME}.zip

DOCKER_RUN=docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE}

download:
	${DOCKER_RUN} /bin/bash -c "test -f ${TRAIN_CSV} || (cd data && curl ${DOWNLOAD_URL} > data.zip && unzip data.zip && rm data.zip)"

train:
	${DOCKER_RUN} python3 main.py --mode ${TRAIN_MODE} --train-csv ${TRAIN_CSV} --model-dir ${MODEL_DIR}

predict:
	${DOCKER_RUN} python3 main.py --test-csv ${TEST_CSV} --prediction-csv ${PREDICTIONS_CSV} --model-dir ${MODEL_DIR}

score:
	${DOCKER_RUN} python3 score.py

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${IMAGE}

run-bash:
	${DOCKER_RUN} /bin/bash

run-jupyter:
	docker run --rm -it -v ${PWD}:/app -w /app -p 8888:8888 ${IMAGE} jupyter notebook --ip=0.0.0.0 --no-browser --allow-root  --NotebookApp.token='' --NotebookApp.password=''

submission:
	${DOCKER_RUN} /bin/bash -c "sed -i.bak 's~{image}~${IMAGE}~g' metadata.json && zip -9 -r submissions/${SUBMISSION_FILE} main.py lib/*.py metadata.json && mv metadata.json.bak metadata.json"
