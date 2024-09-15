# import unittest
# import torch
#
# from zae_engine.models import UserIdModel
#
#
# class TestUserIdModel(unittest.TestCase):
#     def setUp(self):
#         self.d_head = 32
#         self.d_model = 128
#         self.n_head = 8
#         self.num_layers = 6
#         self.num_classes = 1000
#         self.model = UserIdModel(
#             d_head=self.d_head,
#             d_model=self.d_model,
#             n_head=self.n_head,
#             num_layers=self.num_layers,
#             num_classes=self.num_classes,
#         )
#         self.batch_size = 16
#         self.seq_len = 10
#         self.event_vecs = torch.randn(self.batch_size, self.seq_len, self.d_head)
#         self.time_vecs = torch.randint(0, 512, (self.batch_size, self.seq_len))
#         self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
#
#     def test_forward(self):
#         logits, features = self.model(self.event_vecs, self.time_vecs, self.labels)
#         self.assertEqual(logits.size(), (self.batch_size, self.num_classes))
#         self.assertEqual(features.size(), (self.batch_size, self.d_model))
#
#     def test_expand_classes(self):
#         new_num_classes = 1100
#         self.model.expand_classes(new_num_classes)
#         self.assertEqual(self.model.num_classes, new_num_classes)
#         self.assertEqual(self.model.arcface.weight.size(), (new_num_classes, self.d_model))
#
#     def test_forward_after_expand_classes(self):
#         new_num_classes = 1100
#         self.model.expand_classes(new_num_classes)
#         logits, features = self.model(self.event_vecs, self.time_vecs, self.labels)
#         self.assertEqual(logits.size(), (self.batch_size, new_num_classes))
#         self.assertEqual(features.size(), (self.batch_size, self.d_model))
#
#     def test_variable_sequence_length(self):
#         event_vecs = torch.randn(self.batch_size, 5, self.d_head)
#         time_vecs = torch.randint(0, 512, (self.batch_size, 5))
#         logits, features = self.model(event_vecs, time_vecs, self.labels)
#         self.assertEqual(logits.size(), (self.batch_size, self.num_classes))
#         self.assertEqual(features.size(), (self.batch_size, self.d_model))
#
#
# if __name__ == "__main__":
#     unittest.main()
