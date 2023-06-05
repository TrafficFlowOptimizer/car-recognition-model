FROM python:3.9

COPY . .

WORKDIR /src

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "Server:app", "--host", "0.0.0.0", "--port", "8081"]