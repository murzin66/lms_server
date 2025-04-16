FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub[hf_xet]

COPY . .

CMD ["python", "server.py"]


