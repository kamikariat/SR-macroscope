import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
# from data_pytorch import Data

from resnet import MDSR
from data import DIV2K

import numpy as np

import time
import shutil
import yaml
import argparse

# from torchsummary import summary



parser = argparse.ArgumentParser(
    description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
#parser.add_argument('--train', action='store_true')
#parser.add_argument('--data_dir', type=str, required=True)
#parser.add_argument('--image', type=str)
#parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer):
	model.train()
	total_loss = 0
	for i, (input, target) in enumerate(train_loader):
		optimizer.zero_grad()
		input = input.cuda()
		predicted_label = model.forward(input).cuda()
		# target = target.cuda()
		loss = criterion(predicted_label, target).cuda()
		loss.backward()
		optimizer.step()
		total_loss += loss
	return total_loss


def validate(val_loader, model, criterion):
	model.eval()
	with torch.no_grad():
	    total_loss = 0
	    total = 0
	    accuracies = 0
	    for i, (input, target) in enumerate(val_loader):
		    #print(input.shape)
		    #print(target.shape)
		    input = input.cuda()
		    target = target.cuda()
		    predicted_label = model.forward(input).cuda()
		    loss = criterion(predicted_label, target).cuda()
		    total += 128 # todo: change this to dataloader attribute batch_size
		    total_loss += loss
	    return total_loss

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	# best_one stores whether your current checkpoint is better than the previous checkpoint
	if best_one:
		shutil.copyfile(filename, filename2)

def main():
	print(torch.cuda.device_count(), "gpus available")

	n_epochs = config["num_epochs"]
	print("Number of epochs: ", n_epochs)
	model = MDSR()
    
	criterion = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

	train_HR_dataset = './train/x1/'
	train_LR_dataset = './train/x2/'
	dataset = DIV2K(train_HR_dataset, train_LR_dataset)
	train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * .75), int(len(dataset) * .01), int(len(dataset) * .24)])

	train_loader = torch.utils.data.DataLoader(train_dataset, 16)
        val_loader = torch.utils.data.DataLoader(train_dataset, 1)
	#total_loss = train(train_loader, model, criterion, optimizer)
	#print("Total loss", total_loss)


	# todo: val loading
	#val_dataset = './validation/' + '*' #how will you get your dataset
	#val_loader = CIFAR(val_dataset) # how will you use pytorch's function to build a dataloader
	
	
	for epoch in range(n_epochs):
		total_loss = train(train_loader, model, criterion, optimizer)
		print("Epoch {0}: {1}".format(epoch, total_loss))
		validation_loss = validate(val_batches, model, criterion)
		print("Test Loss {0}".format(validation_loss))
		if validation_loss < current_best_validation_loss:
			save_checkpoint(model.state_dict(), True)
			current_best_validation_loss = validation_loss


if __name__ == "__main__":
    main()
