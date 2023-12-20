FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY src src
COPY output output
WORKDIR /src

CMD ["python", "-u", "Server.py"]