import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

from zae-engine.data import load_example
from zae-engine import trainer, models, measure, operation


class ExDataset(data.Dataset):
    def __init__(self, x, y, _type='tuple'):
        self.x = 200 * np.array(x, dtype=float)              # [N, dim]
        self.y = np.array(y)                                # [N, dim]
        self._type = _type

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self._type == 'tuple':
            return torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0),\
                   torch.tensor(self.y[idx], dtype=torch.long)
        elif self._type == 'dict':
            return {'x': torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0),
                    'y': torch.tensor(self.y[idx], dtype=torch.long)}
        else:
            raise ValueError


class ExTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch['x'], batch['y']
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = measure.accuracy(self._to_cpu(y), self._to_cpu(prediction))
        return {'loss': loss, 'output': out, 'acc': acc}

    def test_step(self, batch):
        if isinstance(batch, dict):
            x, y = batch['x'], batch['y']
        elif isinstance(batch, tuple):
            x, y = batch
        else:
            raise TypeError
        out = self.model(x).softmax(1)
        prediction = out.argmax(1)
        loss = F.cross_entropy(out, y)
        acc = measure.accuracy(self._to_cpu(y), self._to_cpu(prediction))
        return {'loss': loss, 'output': out, 'acc': acc}


# class Test_loader(unittest.TestCase):
#     def setUp(self) -> None:
#         self.num_data = 64
#         self.num_class = 4
#         self.device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
#
#     def test_template(self):
#         x, y = load_example()
#         loader = data.DataLoader(dataset=ExDataset(x=[x] * self.num_data, y=[y] * self.num_data), batch_size=8)
#
#         model = models.beat_segmentation(pretrained=True)
#         trainer_ = ExTrainer(model=model, device=self.device, mode='train')
#         res = np.concatenate(trainer_.inference(loader=loader), axis=0).argmax(1)
#
#         res_mat = np.zeros((self.num_class, self.num_class))
#         for o, l in zip(res, loader.dataset.y):
#             res_mat += measure.BijectiveMetric(prediction=o, label=l, num_class=self.num_class).bijective_mat
#
#         operation.print_confusion_matrix(confusion_matrix=res_mat)


if __name__ == '__main__':
    # unittest.main()

    num_data = 40
    num_batch = 7
    num_class = 4
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    ex_x, ex_y = load_example()
    dataset1 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data)
    dataset2 = ExDataset(x=[ex_x] * num_data, y=[ex_y] * num_data, _type='dict')
    ex_loader1 = data.DataLoader(dataset=dataset1, batch_size=num_batch)
    ex_loader2 = data.DataLoader(dataset=dataset2, batch_size=num_batch)

    ex_model = models.beat_segmentation(pretrained=True)
    ex_opt = torch.optim.Adam(params=ex_model.parameters(), lr=1e-2)
    ex_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=ex_opt)

    ex_trainer = ExTrainer(model=ex_model, device=device, mode='train', optimizer=ex_opt, scheduler=ex_scheduler)
    ex_trainer.run(n_epoch=10, loader=ex_loader1, valid_loader=ex_loader2)

    test_y = np.array([ex_y] * num_data)
    test_y[:, 0] = np.arange(num_data) % num_class
    dataset3 = ExDataset(x=[ex_x] * num_data, y=test_y, _type='dict')
    ex_loader3 = data.DataLoader(dataset=dataset3, batch_size=num_batch)

    result = np.concatenate(ex_trainer.inference(loader=ex_loader3), axis=0).argmax(1)
    print(f'Accuracy: {measure.accuracy(test_y, result):.6f}')

    bi_mat = np.zeros((num_class, num_class))
    for o, l in zip(result, ex_loader3.dataset.y):
        bi_mat += measure.BijectiveMetric(prediction=o, label=l, num_class=num_class).bijective_mat

    operation.print_confusion_matrix(confusion_matrix=bi_mat)
    print('Onoff-Accuracy: ', bi_mat.trace() / bi_mat.sum())
