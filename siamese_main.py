import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch_plus import additional_samplers
from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset,GroupedHiCDataset
import HiSiNet.models as models
import torch
from torch_plus.loss import ContrastiveLoss, TripletLoss
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json
import os

parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('learning_rate',  type=float,
                    help='a float for the learning rate')
parser.add_argument('--batch_size',  type=int, default=17,
                    help='an int for batch size')
parser.add_argument('--epoch_training',  type=int, default=30,
                    help='an int for no of epochs training can go on for')
parser.add_argument('--epoch_enforced_training',  type=int, default=0,
                    help='an int for number of epochs to force training for')
parser.add_argument('--outpath',  type=str, default="outputs/",
                    help='a path for the output directory')
parser.add_argument('--seed',  type=int, default=30004,
                    help='an int for the seed')
parser.add_argument('--mask',  type=bool, default=False,
                    help='an argument specifying if the diagonal should be masked')
parser.add_argument('--bias',  type=float, default=2,
                    help='an argument specifying the bias towards the contrastive loss function')
parser.add_argument("data_inputs", nargs='+',help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(n_gpu))
else:
    device = torch.device("cpu")
    print("Using CPU")

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

torch.manual_seed(args.seed)

# Initialize dataset
Siamese = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["training"]],
            reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])
train_sampler = torch.utils.data.RandomSampler(Siamese)

# Initialize CNN parameters
batch_size, learning_rate = args.batch_size, args.learning_rate
no_of_batches = np.floor(len(Siamese) / args.batch_size)
dataloader = DataLoader(Siamese,
                        batch_size=args.batch_size,
                        sampler=train_sampler,
                        num_workers=4,
                        pin_memory=True)

# Create validation dataset and dataloader
Siamese_validation = GroupedHiCDataset(
    [SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["validation"]],
                       reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese_validation)
batches_validation = np.ceil(len(Siamese_validation) / 100)
dataloader_validation = DataLoader(Siamese_validation,
                                   batch_size=100,
                                   sampler=test_sampler,
                                   num_workers=4,
                                   pin_memory=True)

# Initialize the Convolutional neural network
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

model_save_path = args.outpath + args.model_name + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(
    args.seed)
torch.save(model.state_dict(), model_save_path + '.ckpt')

# Initialize classification network
nn_model = models.LastLayerNN()
if torch.cuda.device_count() > 1:
    nn_model = nn.DataParallel(nn_model)
nn_model = nn_model.to(device)
torch.save(nn_model.state_dict(), model_save_path + "_nn.ckpt")

# Initialize loss functions and optimizer
# criterion = ContrastiveLoss()
criterion = TripletLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters())

# Training loop
for epoch in range(args.epoch_training):
    model.train()
    nn_model.train()
    running_loss1, running_loss2 = 0.0, 0.0
    running_validation_loss = 0.0

    # Training phase
    for i, data in enumerate(dataloader):
        input1, input2, labels = data
        input1, input2 = input1.to(device), input2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output1, output2 = model(input1, input2)
        output_class = nn_model(output1, output2)

        loss2 = criterion2(output_class, labels)
        labels = labels.type(torch.FloatTensor).to(device)
        loss1 = criterion(output1, output2, labels)
        loss = args.bias * loss1 + loss2

        loss.backward()
        optimizer.step()

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()

        if (i + 1) % no_of_batches == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss1: {:.4f}, Loss2: {:.4f}'.format(
                epoch + 1,
                args.epoch_training,
                i + 1,
                int(no_of_batches),
                running_loss1 / no_of_batches,
                running_loss2 / no_of_batches
            ))

    # Validation phase
    model.eval()
    nn_model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_validation):
            input1, input2, labels = data
            input1, input2 = input1.to(device), input2.to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, labels)
            running_validation_loss += loss.item()

    print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(
        epoch + 1,
        args.epoch_training,
        running_validation_loss / batches_validation
    ))

    # Early stopping check
    if epoch > args.epoch_enforced_training:
        prev_validation_loss = min(prev_validation_loss, running_validation_loss)
        if float(prev_validation_loss) < 1.1 * float(running_validation_loss):
            print("Early stopping triggered")
            break
    else:
        prev_validation_loss = running_validation_loss

    # Save model checkpoints
    torch.save(model.state_dict(), model_save_path + '.ckpt')
    torch.save(nn_model.state_dict(), model_save_path + "_nn.ckpt")

print("Training completed")