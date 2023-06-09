FROM python:3.7-slim-buster
USER root
WORKDIR /app
COPY main.py requirements.txt utilities.py /app/
COPY static /app/static
COPY templates /app/templates
COPY data  /app/data
COPY m  /app/model
RUN pip install  -r requirements.txt
EXPOSE 5000
CMD ["gunicorn"  , "--bind", "0.0.0.0:5000", "main:app"]