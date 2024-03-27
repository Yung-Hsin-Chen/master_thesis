FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip \
    &&  pip3 install --requirement requirements.txt

COPY . .

CMD [ "python" ]