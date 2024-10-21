import grpc
from concurrent import futures
import time
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

import reranker_service_pb2
import reranker_service_pb2_grpc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = '/app/model'


class RerankerService(reranker_service_pb2_grpc.RerankerServiceServicer):
    def __init__(self):
        logger.info("Loading tokenizer and model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH)
            self.model.eval()
            logger.info("Tokenizer and model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def RerankDocuments(self, request, context):
        start_time = time.time()
        logger.info(
            f"Received rerank request for query: {request.query[:50]}...")
        try:
            pairs = [[request.query, doc] for doc in request.documents]
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    return_tensors='pt', max_length=512)
            with torch.no_grad():
                scores = self.model(
                    **inputs, return_dict=True).logits.view(-1,).float()

            scored_documents = sorted(
                zip(request.documents, scores.tolist()), key=lambda x: x[1], reverse=True)

            reranked_documents = [
                reranker_service_pb2.RankedDocument(document=doc, score=score)
                for doc, score in scored_documents
            ]

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(
                f"Reranking completed successfully. Processing time: {processing_time:.2f} seconds")

            return reranker_service_pb2.RerankResponse(reranked_documents=reranked_documents)
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return reranker_service_pb2.RerankResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    reranker_service_pb2_grpc.add_RerankerServiceServicer_to_server(
        RerankerService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
