# Sberbank Data Science Journey 2018: Docker-friendly baseline

Для работы необходим `make` и `docker`. Перед началом работы нужно скачать датасет в папку `data`.

## Make-команды для работы с Docker :whale:

`make download` - cкачать датасет в папку data.  
`make train DATASET=1` - обучение модели на датасете с указанным номером [1-8].  
`make predict DATASET=1` - валидация модели на датасете с указанным номером [1-8].  
`make score` - валидация модели на всех датасетах и сохранение результата в папку scores.  
`make docker-build` - сборка Docker-образа.  
`make docker-push` - залить Docker-образ на Docker Hub.  
`make run-bash` - запустить терминал в Docker-контейнере.  
`make run-jupyter` - запустить Jupyter в Docker-контейнере по адресу http://localhost:8888.  
`make submission` - создать сабмит-файл в директории submissions.  

## Сборка своего Docker-образа

1. Отредактировать `Makefile` и указать название образа на первой строчке `IMAGE=username/image`.  
2. Отредактировать `Dockerfile`.
3. Запустить сборку образа `make docker-build`.
4. Залить Docker-образ на Docker Hub `make docker-push`.

## Запуск на Windows (в процессе)

1. Установить [Docker](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe).
2. Установить [Make](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe). И добавить его в Path.
3. Запустить Docker. В настройках выделить докеру нужное количество оперативной памяти.
4. Запустить PowerShell от имени администратора. При выполнении make-комманд дать разрешение на монтирование директории для докера.
