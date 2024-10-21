# Reranker Service

This project implements a gRPC-based reranker service using the BAAI/bge-reranker-v2-m3 model. It provides an efficient way to rerank a list of documents based on their relevance to a given query.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- gRPC-based reranker service
- Uses the BAAI/bge-reranker-v2-m3 model for document reranking
- Docker support for easy deployment
- Efficient inference using PyTorch

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/grpc-reranker-service.git
   cd grpc-reranker-service
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Generate gRPC code from the proto file:
   ```
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. reranker_service.proto
   ```

## Usage

To start the gRPC server:

```
python main.py
```

The server will start and listen on port 50051.

To use the service, you need to create a gRPC client. Here's a simple example:

```python
import grpc
import reranker_pb2
import reranker_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = reranker_pb2_grpc.RerankerServiceStub(channel)
        request = reranker_pb2.RerankRequest(
            query="What is the capital of France?",
            documents=[
                "Paris is the capital of France.",
                "London is the capital of the United Kingdom.",
                "Berlin is the capital of Germany."
            ]
        )
        response = stub.RerankDocuments(request)
        print("Reranked documents:")
        for doc in response.reranked_documents:
            print(f"Score: {doc.score:.4f}, Document: {doc.document}")

if __name__ == '__main__':
    run()
```

## API Reference

The service provides a single RPC method:

- `RerankDocuments(RerankRequest) returns (RerankResponse)`
  - Input:
    - `query` (string) - The query to rank documents against
    - `documents` (repeated string) - The list of documents to be reranked
  - Output:
    - `reranked_documents` (repeated RankedDocument) - The reranked documents with their scores

## Docker Deployment

To build and run the service using Docker:

1. Build the Docker image:
   ```
   docker-compose build
   ```

2. Start the service:
   ```
   docker-compose up
   ```

The service will be available on the host machine at `localhost:50051`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the BAAI/bge-reranker-v2-m3 model. We acknowledge the authors for their work in creating and open-sourcing this model.
