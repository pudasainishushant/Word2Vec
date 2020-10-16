FROM python:3.6
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get update -y
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python nltk_download.py
CMD streamlit run main.py --server.port 8051