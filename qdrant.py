from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://5617597b-eac0-43bf-9426-ac44034829b7.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Gw3K57yWcyRjDHs63fzD1psr2FLYqy4TLiYIG7JWH4U",
)

print(qdrant_client.get_collections())