FROM apache/airflow:2.9.3

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY dags.py /opt/airflow/dags
COPY src /opt/airflow/src

CMD ["airflow", "standalone"]
