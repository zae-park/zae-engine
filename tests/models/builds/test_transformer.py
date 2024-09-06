import unittest
import torch

from zae_engine.models import TransformerBase, EncoderBase, DecoderBase


class TestTransformerBase(unittest.TestCase):
    def setUp(self):
        # 설정: 모델의 하이퍼파라미터와 입력 데이터 정의
        self.d_model = 768
        self.num_layers = 4
        self.src_vocab_size = 1024
        self.tgt_vocab_size = 1024
        self.max_len = 256

        # 임베딩 레이어
        self.src_emb = torch.nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_emb = torch.nn.Embedding(self.tgt_vocab_size, self.d_model)

        # 인코더 및 디코더 정의
        self.encoder = EncoderBase(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_factory=torch.nn.TransformerEncoderLayer,
            dim_feedforward=512,
            batch_first=True,
        )
        self.decoder = DecoderBase(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_factory=torch.nn.TransformerDecoderLayer,
            dim_feedforward=512,
            batch_first=True,
        )

        # TransformerBase 정의
        self.model = TransformerBase(
            encoder_embedding=self.src_emb, decoder_embedding=self.tgt_emb, encoder=self.encoder, decoder=self.decoder
        )

        # 샘플 데이터 생성
        self.src = torch.randint(0, self.src_vocab_size, (32, self.max_len))  # batch_size=32, seq_len=max_len
        self.tgt = torch.randint(0, self.tgt_vocab_size, (32, self.max_len))

    def test_forward(self):
        # 기본적인 forward 패스를 테스트
        output = self.model(self.src, self.tgt)
        self.assertEqual(output.size(), (32, self.max_len, self.d_model))  # 예상 출력 크기

    def test_encoder_only(self):
        # 디코더 없이 인코더만으로 모델을 테스트
        model_encoder_only = TransformerBase(encoder_embedding=self.src_emb, encoder=self.encoder, decoder=None)
        output = model_encoder_only(self.src)
        self.assertEqual(output.size(), (32, self.max_len, self.d_model))

    def test_variable_sequence_length(self):
        # 다양한 시퀀스 길이를 사용한 테스트
        src_var_len = torch.randint(0, self.src_vocab_size, (32, 256))  # 짧은 시퀀스
        tgt_var_len = torch.randint(0, self.tgt_vocab_size, (32, 256))
        output = self.model(src_var_len, tgt_var_len)
        self.assertEqual(output.size(), (32, 256, self.d_model))

    def test_with_mask(self):
        # 마스크를 사용한 테스트
        src_mask = torch.ones(32, self.max_len).bool()
        tgt_mask = torch.ones(32, self.max_len).bool()
        output = self.model(self.src, self.tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        self.assertEqual(output.size(), (32, self.max_len, self.d_model))


if __name__ == "__main__":
    unittest.main()
