FROM python:3.8-alpine

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev g++ \
    libffi-dev openssl-dev \
    libxml2 libxml2-dev \
    libxslt libxslt-dev \
    libjpeg-turbo-dev zlib-dev

RUN pip install --upgrade cython

RUN pip install --upgrade pip

EXPOSE 5000/tcp

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app.py .
COPY prediction_engine.py .
COPY train.csv .

CMD ["python", "./app.py"]
