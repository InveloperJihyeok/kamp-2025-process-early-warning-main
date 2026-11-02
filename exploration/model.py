import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    """
    제품 임베딩을 사용하는 통합 신경망 모델 아키텍처 (PyTorch 버전)
    """
    def __init__(self, num_unique_products, embedding_dim, num_other_features):
        super(EmbeddingModel, self).__init__()
        
        # 1. 임베딩 층: 제품 인덱스를 받아 임베딩 벡터로 변환
        self.product_embedding = nn.Embedding(
            num_embeddings=num_unique_products,
            embedding_dim=embedding_dim
        )
        
        # 2. 은닉층 (Dense Layers): 임베딩 벡터와 나머지 피처를 합쳐 처리
        # 입력 차원: 임베딩 차원 + 나머지 피처의 개수
        input_dim = embedding_dim + num_other_features
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 3. 출력층: 최종 수주량 예측
        self.output_layer = nn.Linear(64, 1)

    def forward(self, product_id_input, other_features_input):
        """순전파 로직을 정의"""
        
        # 제품 ID를 임베딩 벡터로 변환
        # (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim)
        embedded_product = self.product_embedding(product_id_input).squeeze(1)
        
        # 임베딩 벡터와 나머지 피처 벡터를 결합
        concatenated = torch.cat([embedded_product, other_features_input], dim=1)
        
        # 은닉층과 출력층을 통과
        hidden_output = self.hidden_layers(concatenated)
        final_output = self.output_layer(hidden_output)
        
        return final_output