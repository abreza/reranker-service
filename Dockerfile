FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model

RUN wget -O /app/model/config.json https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/config.json
RUN wget -O /app/model/model.safetensors https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/model.safetensors
RUN wget -O /app/model/sentencepiece.bpe.model https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/sentencepiece.bpe.model
RUN wget -O /app/model/special_tokens_map.json https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/special_tokens_map.json
RUN wget -O /app/model/tokenizer.json https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer.json
RUN wget -O /app/model/tokenizer_config.json https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer_config.json

COPY . .

EXPOSE 50051

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]