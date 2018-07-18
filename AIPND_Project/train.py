import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import json
from collections import OrderedDict
import os    


def train_model(model,train_dataloaders,valid_dataloaders,optimizer,criterion,device,epochs):

    print_every = 100
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        for (inputs, labels) in train_dataloaders:
            steps += 1
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            #print(type(inputs))
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_dataloaders, criterion,device)
                print("Epoch: {}/{}... ".format(e+1, epochs))
                print("Train Loss: {:.4f}".format(running_loss/print_every))
                print("Validation Loss: {:.4f}".format(test_loss/len(valid_dataloaders)))
                print("Validation Accuracy: {:.4f}".format(accuracy/len(valid_dataloaders)))
            
                running_loss = 0
                # Make sure training is back on
                model.train()
            
def validation(model, valid_dataloaders, criterion,device):
    test_loss = 0
    accuracy = 0
    for images, labels in valid_dataloaders:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        if torch.cuda.is_available():
            equality = equality.type(torch.cuda.FloatTensor)
        accuracy += equality.mean()
    
    return test_loss, accuracy

def save_model(model,optimizer,traindata,epochs,lr,architecture,hidden_units):
    checkpoint = {'architecture':architecture,
                  'hidden units':hidden_units,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'mapping':traindata.class_to_idx,
                  'num_epoch':epochs,
                 'learning_rate':lr}

    torch.save(checkpoint, 'model/checkpoint.pth')

def set_model(hidden_layers,architecture,learning_rate):

    if architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = model.classifier[1].in_features
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = model.classifier.in_features
    elif architecture == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        input_size = model.fc.in_features
    else:
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    
    #print(model)
    #input_size = model.classifier[0].in_features
    #print(model)
    #print(input_size)
    for param in model.parameters():
        param.requires_grad = False



    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, hidden_layers)),
                          ('relu2', nn.ReLU()),
                          ('dropout2',nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(hidden_layers, 392)),
                          ('relu3', nn.ReLU()),
                          ('dropout3',nn.Dropout(p=0.2)),
                          ('fc4', nn.Linear(392, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    if architecture == 'resnet18' or architecture == 'inception_v3':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model,criterion,optimizer
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,help="Directory of the model",default='../aipnd-project/flowers')
    parser.add_argument('--save_dir',type=str,help="Directory to save the model",default='./model')
    parser.add_argument('--architecture',type=str,help="architecture of the model",default='vgg16')
    parser.add_argument('--learning_rate',type=float,help="learning rate of the model",default=0.001)
    parser.add_argument('--hidden_units',type=int,help="Hidden Units of the model",default=1024)
    parser.add_argument('--epochs',type=int,help="Epochs the model",default=10)
    parser.add_argument('--device_type',type=str,help="Device used by model",default='gpu')
    args = parser.parse_args()
    #print(args.data_directory)
    #print(args.learning_rate)
    #print(args.hidden_units)
    #print(args.epochs)
    #print(args.device_type)
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])]),
                       'valid':transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                    [0.229, 0.224, 0.225])]),
                       'test':transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                  [0.229, 0.224, 0.225])])
                      }
    
    data_dir = args.data_directory
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device_type == 'gpu') else "cpu")
    model,criterion,optimizer = set_model(args.hidden_units,args.architecture,args.learning_rate)
    model.to(device)
    num_epochs = args.epochs;
    train_model(model,dataloaders['train'],dataloaders['valid'],optimizer,criterion,device,num_epochs)
    save_model(model,optimizer,image_datasets['train'],num_epochs,args.learning_rate,args.architecture,args.hidden_units)
   
if __name__ == '__main__':
    main()

