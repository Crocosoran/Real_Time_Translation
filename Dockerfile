FROM python:3.8.3-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y fontconfig && \
    mkdir -p /usr/share/fonts/truetype/custom



RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system

COPY . .

COPY ./fonts/STHeiti_Medium.ttc /usr/share/fonts/truetype/custom/

RUN fc-cache -fv

ENV PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "Streamlit/app.py"]

#FROM python:3.8.3-slim
#
## Set the working directory
#WORKDIR /app
#
## Copy the requirements file and install dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy the rest of the application code
#COPY . .
#
## Expose the port your Streamlit or FastAPI app runs on (example: 8501 or 8000)
#EXPOSE 8501
#
## Command to run the application (update this as needed)
#CMD ["streamlit", "run", "Streamlit/app.py"]
