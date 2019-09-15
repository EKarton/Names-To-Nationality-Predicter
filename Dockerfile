FROM python:3

COPY . /app

RUN pip3 install -r "/app/Web App/requirements.txt"
RUN pip3 install -r "/app/ML Model/requirements.txt"

WORKDIR "/app/Web App"

EXPOSE 5000

CMD ["python3", "app.py"]