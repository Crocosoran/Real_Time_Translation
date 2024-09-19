FROM python:3.8.3-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y fontconfig && \
    mkdir -p /usr/share/fonts/truetype/custom



RUN pip install --no-cache-dir pipenv

COPY ../Pipfile Pipfile.lock ./

RUN pipenv install --system

COPY fonts /app/fonts
COPY saved_models/quantized_model /app/saved_models/quantized_model
COPY src/source_vectorisation /app/src/source_vectorisation
COPY src/target_vectorisation /app/src/target_vectorisation
COPY src/data_processing.py /app/src/
COPY Streamlit /app/Streamlit

COPY ../fonts/STHeiti_Medium.ttc /usr/share/fonts/truetype/custom/

RUN fc-cache -fv

ENV PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "Streamlit/app.py"]


