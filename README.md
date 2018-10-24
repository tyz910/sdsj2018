# Sberbank Data Science Journey 2018: Docker-friendly baseline

Для работы необходим `make` и `docker`. Перед началом работы нужно скачать датасет в папку `data`.

## Особенности решения

* LightGBM с подбором гиперпараметров через hyperopt.
* Mean target Encoding для категориальных фич.
* Для 8-го датасета отбор фич через [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py).
* Лик от [bagxi](https://github.com/bagxi/sdsj2018_lightgbm_baseline).

Так же есть, но не используются: Vowpal Wabbit, H2O AutoML.
Скор на ЛБ: `5,30072`.

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

1. Зарегистрироваться на [Docker Hub](https://hub.docker.com/).
2. Отредактировать [Makefile](https://github.com/tyz910/sdsj2018/blob/master/Makefile) и указать название образа на первой строчке `IMAGE=username/image`.  
3. Отредактировать [Dockerfile](https://github.com/tyz910/sdsj2018/blob/master/Dockerfile) и добавить установку нужных пакетов.
4. Запустить сборку образа `make docker-build`.
5. Залить Docker-образ на Docker Hub `make docker-push`.
6. Убедиться, что созданный репозиторий публичный (Public), а не приватный (Private). Приватность настраивается по ссылке `https://hub.docker.com/r/username/image/~/settings/`.

## Запуск на Windows

1. Установить [Docker](https://download.docker.com/win/stable/Docker%20for%20Windows%20Installer.exe).
2. Установить [Make](http://gnuwin32.sourceforge.net/downlinks/make.php). И добавить его в [PATH](https://ru.stackoverflow.com/questions/153628/Как-добавить-путь-в-переменную-окружения-path-на-windows).
3. Запустить Docker. В настройках выделить докеру нужное количество оперативной памяти.
4. Запустить PowerShell от имени администратора. При выполнении make-команд дать разрешение на монтирование директории для докера.
