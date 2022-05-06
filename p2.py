import random
import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

train_data = scio.loadmat('MNIST_data/train_data.mat')
test_data = scio.loadmat('MNIST_data/test_data.mat')
Xtrain, Ytrain = train_data['X'], train_data['Y']
Xtest, Ytest = test_data['X'], test_data['Y']

train_dataset = TensorDataset(torch.tensor(Xtrain.astype(np.float32)), torch.tensor(Ytrain.astype(np.int64)))
test_dataset = TensorDataset(torch.tensor(Xtest.astype(np.float32)), torch.tensor(Ytest.astype(np.int64)))
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1000)

class MLP(nn.Module):
	def __init__(self, in_f, hidden_f, out_f):
		super(MLP, self).__init__()
		self.linear1 = nn.Linear(in_features=in_f, out_features=hidden_f)
		self.linear2 = nn.Linear(in_features=hidden_f, out_features=out_f)

	def forward(self, x):
		return self.linear2(F.relu(self.linear1(x)))

	def init_params(self, method='kaiming_normal'):
		if method == 'kaiming_normal':
			for m in self.modules():
				if isinstance(m, nn.Linear):
					nn.init.kaiming_normal_(m.weight.data)
					nn.init.constant_(m.bias.data, 0)
		elif method == 'xavier_uniform':
			for m in self.modules():
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight.data)
					nn.init.constant_(m.bias.data, 0)
		elif method == 'xavier_normal':
			for m in self.modules():
				if isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight.data)
					nn.init.constant_(m.bias.data, 0)
		else:
			raise ValueError

def train_MLP(model):
	optimizer = optim.Adam(params=model.parameters(), lr=0.001)
	crit = nn.CrossEntropyLoss()

	train_loss = []
	test_loss = []
	train_acc = []
	test_acc = []

	for epoch in range(40):
		model.train()
		train_batch_loss = 0.0
		train_batch_tp = 0.0

		for batch in train_dataloader:
			X = batch[0]
			Y = batch[1].view(-1)
			model.zero_grad()

			logits = model(X)
			loss = crit(logits, Y)
			loss.backward()
			optimizer.step()

			train_batch_loss += loss.item()
			pred = torch.argmax(logits, dim=1)
			train_batch_tp += torch.sum(pred == Y).item()

		train_loss.append(train_batch_loss / len(train_dataloader))
		train_acc.append(train_batch_tp / len(Xtrain))

		test_batch_loss = 0.0
		test_batch_tp = 0.0

		for batch in test_dataloader:
			X = batch[0]
			Y = batch[1].view(-1)

			logits = model(X)
			loss = crit(logits, Y)

			test_batch_loss += loss.item()
			test_batch_tp += torch.sum(torch.argmax(logits, dim=1) == Y).item()

		test_loss.append(test_batch_loss / len(test_dataloader))
		test_acc.append(test_batch_tp / len(Xtest))

		# print(f'Epoch {epoch} Training Loss:', train_loss[-1])
		# print(f'Epoch {epoch} Training Acc:', train_acc[-1])
		# print(f'Epoch {epoch} Testing Loss:', test_loss[-1])
		# print(f'Epoch {epoch} Testing Acc:', test_acc[-1], '\n')

	return train_loss, test_loss, train_acc, test_acc

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for trial in range(3):
	print(f'==========Trial {trial}==========')
	model = MLP(784, 32, 10)
	if trial == 0:
		model.init_params('kaiming_normal')
	elif trial == 1:
		model.init_params('xavier_uniform')
	else:
		model.init_params('xavier_normal')
	
	train_loss, test_loss, train_acc, test_acc = train_MLP(model)
	train_losses.append(train_loss)
	test_losses.append(test_loss)
	train_accs.append(train_acc)
	test_accs.append(test_acc)

	print("Train Acc:{:.2f}\tTest Acc:{:.2f}\t".format(train_acc[-1] * 100, test_acc[-1] * 100))

train_loss_avg = np.mean(train_losses, axis=0)
test_loss_avg = np.mean(test_losses, axis=0)
train_acc_avg = np.mean(train_accs, axis=0)
test_acc_avg = np.mean(test_accs, axis=0)

train_loss_std = np.std(train_losses, axis=0)
test_loss_std = np.std(test_losses, axis=0)
train_acc_std = np.std(train_accs, axis=0)
test_acc_std = np.std(test_accs, axis=0)

x_axis = np.arange(40)

plt.errorbar(x_axis, train_loss_avg, yerr=train_loss_std, label='Training')
plt.errorbar(x_axis, test_loss_avg, yerr=test_loss_std, label='Testing')
plt.xlabel('#Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.pdf')

plt.clf()

plt.errorbar(x_axis, train_acc_avg, yerr=train_acc_std, label='Training')
plt.errorbar(x_axis, test_acc_avg, yerr=test_acc_std, label='Testing')
plt.xlabel('#Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Acc.pdf')




