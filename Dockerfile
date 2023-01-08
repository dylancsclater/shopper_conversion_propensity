FROM python:3.8

COPY requirements.txt /xgboost/requirements.txt

RUN pip install -r /xgboost/requirements.txt

COPY model.py /xgboost/model.py

CMD ["python", "/app/model.py"]
