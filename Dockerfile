FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -e .
