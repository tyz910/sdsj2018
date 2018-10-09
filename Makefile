IMAGE=tyz910/sdsj2018

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

DOWNLOAD_URL=https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip

download:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} test -f ${TRAIN_CSV} || \
	(cd data && curl ${DOWNLOAD_URL} > data.zip && unzip data.zip && rm data.zip)

train:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} python3 main.py --mode ${TRAIN_MODE} --train-csv ${TRAIN_CSV} --model-dir ${MODEL_DIR}

predict:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} python3 main.py --test-csv ${TEST_CSV} --prediction-csv ${PREDICTIONS_CSV} --model-dir ${MODEL_DIR}

score:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} python3 score.py

docker-build:
	docker build -t ${IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images

docker-push:
	docker push ${IMAGE}

run-bash:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} /bin/bash

run-jupyter:
	(sleep 3 && python -mwebbrowser http://localhost:8888) & docker run --rm -it -v ${PWD}:/app -w /app -p 8888:8888 ${IMAGE} jupyter notebook --ip=0.0.0.0 --no-browser --allow-root  --NotebookApp.token='' --NotebookApp.password=''

submission:
	docker run --rm -it -v ${PWD}:/app -w /app ${IMAGE} sed -i.bak 's~{image}~${IMAGE}~g' metadata.json && \
	zip -9 -r submissions/${SUBMISSION_FILE} main.py lib/*.py metadata.json && mv metadata.json.bak metadata.json
