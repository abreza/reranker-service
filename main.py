import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import time
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = '/app/model'
logger.info("Loading tokenizer and ONNX model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Tokenizer and model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
async def rerank_documents(request: RerankRequest):
    start_time = time.time()
    logger.info(f"Received rerank request for query: {request.query[:50]}...")
    try:
        pairs = [[request.query, doc] for doc in request.documents]

        inputs = tokenizer(pairs, padding=True, truncation=True,
                           return_tensors='pt', max_length=512)

        with torch.no_grad():
            scores = model(**inputs, return_dict=True).logits.view(-1,).float()

        scored_documents = sorted(
            zip(request.documents, scores.tolist()), key=lambda x: x[1], reverse=True)

        result = {
            "reranked_documents": [
                {"document": doc, "score": score} for doc, score in scored_documents
            ]
        }

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(
            f"Reranking completed successfully. Processing time: {processing_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting reranker microservice...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
