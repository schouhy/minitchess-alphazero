from erlyx.learners import BaseLearner
from torch.utils.data import DataLoader
from exp.environment import MinitChessEpisode, NUM_ACTIONS
from exp.dataset import SimpleAlphaZeroDataset
import numpy as np
import torch


def collate_fn(batch):
    pis = []
    for item in batch:
        pi = np.zeros(NUM_ACTIONS)
        pi[item['legal_moves']] = item['pi']
        pis.append(pi)
    obs = [
        MinitChessEpisode(o['observation']).get_board_array()[None]
        for o in batch
    ]
    rewards = [o['reward'] for o in batch]
    return map(
        torch.FloatTensor,
        [np.vstack(pis), np.vstack(obs),
         np.vstack(rewards)])


class SimpleAlphaZeroLearner(BaseLearner):
    def __init__(self, policy, batch_size, learning_rate):
        self._policy = policy
        self._batch_size = batch_size
        self.optimizer = torch.optim.AdamW(policy.model.parameters(),
                                           lr=learning_rate)
        self._loss_file = 'learner_losses.txt'
        self._counter = 0

    def update(self, dataset: SimpleAlphaZeroDataset):
        dataloader = DataLoader(dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        model = self._policy.model.train().cuda()

        with open(self._loss_file, 'a') as outfile:
            outfile.write('### Training begin ###\n')
            for pib, boardb, reward in iter(dataloader):
                pib, boardb, reward = pib.cuda(), boardb.cuda(), reward.cuda()
                pb, vb = model(boardb)
                pb = pb.log_softmax(-1) 
                loss = ((vb - reward)**2 - (pib * pb).sum(1)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                outfile.write(f'loss: {loss}\n')

        torch.save(model.state_dict(), f'weights_update_{self._counter}')
        self._counter += 1
