FROM python:3.11-bookworm

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY dataset /opt/dataset
COPY src /opt/src

WORKDIR /opt
