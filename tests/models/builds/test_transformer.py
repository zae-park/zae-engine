import unittest
import torch
import torch.nn as nn

from zae_engine.models import TransformerBase, BertBase, EncoderBase, DecoderBase


class TestTransformerBase(unittest.TestCase):
    def setUp(self):
        # 설정: 모델의 하이퍼파라미터와 입력 데이터 정의
        self.d_model = 256
        self.d_ff = 64
        self.num_layers = 2
        self.src_vocab_size = 128
        self.tgt_vocab_size = 128
        self.max_len = 64

        # 임베딩 레이어
        self.src_emb = torch.nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_emb = torch.nn.Embedding(self.tgt_vocab_size, self.d_model)

        # 인코더 및 디코더 정의
        self.encoder = EncoderBase(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_factory=torch.nn.TransformerEncoderLayer,
            dim_feedforward=self.d_ff,
            batch_first=True,
        )
        self.decoder = DecoderBase(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_factory=torch.nn.TransformerDecoderLayer,
            dim_feedforward=self.d_ff,
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
        src_key_padding_mask = torch.zeros(32, self.max_len).bool()  # 패딩 마스크는 0
        tgt_key_padding_mask = torch.zeros(32, self.max_len).bool()

        # Attention mask: Future positions을 가리지 않음 (디코더에서 사용)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.max_len).bool()

        # 모델에 마스크 입력
        output = self.model(
            self.src,
            self.tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # 출력 크기 확인
        self.assertEqual(output.size(), (32, self.max_len, self.d_model))


class TestBertBase(unittest.TestCase):
    def setUp(self):
        # Define model hyperparameters
        self.d_model = 256
        self.d_ff = 64
        self.num_layers = 2
        self.max_len = 64
        self.batch_size = 16

        # Define encoder (no embedding layer)
        self.encoder = EncoderBase(
            d_model=self.d_model,
            num_layers=self.num_layers,
            layer_factory=nn.TransformerEncoderLayer,
            dim_feedforward=self.d_ff,
            batch_first=True,
        )

        # Define BertBase model (encoder_embedding is nn.Identity since input_embeds are precomputed)
        self.model = BertBase(encoder_embedding=nn.Identity(), encoder=self.encoder, dim_hidden=self.d_model)

        # Generate precomputed embeddings (batch_size=32, seq_len=512, embedding_dim=768)
        self.input_embeds = torch.randn(self.batch_size, self.max_len, self.d_model)

    def test_forward_with_pooler(self):
        # Test forward pass with precomputed embeddings and a pooler
        output = self.model(self.input_embeds)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))  # Check expected output size

    def test_forward_without_pooler(self):
        # Test forward pass without a pooler, using precomputed embeddings
        model_without_pooler = BertBase(encoder_embedding=nn.Identity(), encoder=self.encoder)
        output = model_without_pooler(self.input_embeds)
        self.assertEqual(output.size(), (self.batch_size, self.max_len, self.d_model))  # Check expected output size

    def test_variable_sequence_length(self):
        # Test with variable sequence lengths using precomputed embeddings
        input_embeds_var_len = torch.randn(self.batch_size, 12, self.d_model)  # Shorter sequence
        output = self.model(input_embeds_var_len)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))  # Check pooler output size

    def test_with_mask(self):
        # Test with masking applied using precomputed embeddings
        src_mask = torch.ones(self.max_len, self.max_len).bool()
        output = self.model(self.input_embeds, src_mask=src_mask)
        self.assertEqual(output.size(), (self.batch_size, self.d_model))  # Check pooler output size

    # def test_token_type_ids_generation(self):
    #     # Test token_type_ids generation based on [SEP] tokens
    #     src_with_sep = torch.randint(0, self.src_vocab_size, (32, self.max_len)).tolist()
    #
    #     for i in range(32):
    #         # Ensure only one [SEP] token is inserted per sequence to avoid errors
    #         src_with_sep[i][self.max_len // 2] = self.model.sep_token_id  # Insert a single [SEP] token
    #
    #     token_type_ids = self.model._generate_token_type_ids(src_with_sep)
    #
    #     # Check that the token_type_ids are correctly generated
    #     self.assertTrue(torch.all(token_type_ids[:, : self.max_len // 2] == 0))  # First part should be 0
    #     self.assertTrue(torch.all(token_type_ids[:, self.max_len // 2 + 1 :] == 1))  # Second part should be 1


if __name__ == "__main__":
    unittest.main()
