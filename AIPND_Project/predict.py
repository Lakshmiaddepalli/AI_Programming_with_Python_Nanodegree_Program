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
import json
import train
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    #model = fc_model.Network(checkpoint['input_size'],checkpoint['output_size'],checkpoint['hidden_layers'])
    model,criterion,optimizer = train.set_model(checkpoint['hidden units'],checkpoint['architecture'],checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    idx_to_class = dict()
    for key,value in model.class_to_idx.items():
        idx_to_class[value] = key
    #print(idx_to_class)
    return model,criterion,optimizer,idx_to_class

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    width, height = im.size
    #print(width,height)
    size = 256,(height*256)/width
    if(width > height):
        size = (width*256)/height,256
    im.thumbnail(size)
    width, height = im.size
    #print(width,height)
    new_width = 224;
    new_height=224;
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))
    width, height = im.size
    #print(width,height)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]


    np_image = np.array(im)
    np_image_min = np_image.min(axis=(0, 1), keepdims=True)
    np_image_max = np_image.max(axis=(0, 1), keepdims=True)
    np_image = (np_image - np_image_min)/(np_image_max-np_image_min)
    #np_image = np_image/255
    normalised_image=(np_image-mean)/std
    normalised_image.transpose(2,0,1)
    
    return normalised_image.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #print(image.shape)
    image = image.transpose((1, 2, 0))
    #print(image.shape)
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, num_classes):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #print(image_path.shape)
    model.eval()
    outputs = model(image_path)
    ps = torch.exp(outputs).data
    #print(cat_to_name)
    #print(ps.shape)
    return torch.topk(input=ps, k=num_classes)


def sanity_check(image_path,device,model,category_file_location,num_classes,idx_to_class):
    img_path=image_path
    #image = Image.open(img_path)
    nmp_tensor = process_image(img_path)
    img_tensor = torch.from_numpy(nmp_tensor)
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        img_tensor = img_tensor.type(torch.FloatTensor)
    image = img_tensor.unsqueeze(0) 
    #image = image.to(device)
    #print(type(img_tensor))
    #print(type(image))

    probs, classes = predict(image,model,num_classes)
    class1 = classes.cpu().numpy()
    prob = probs.cpu().numpy()
    #print(type(prob))
    #print((prob))
    #print(np.max(prob))
    class_names = []
    for x in np.nditer(class1):
        #print(int(x))
        class_value = idx_to_class.get(int(x))
        #print(class_value)
        #print(category_file_location)
        #print(category_file_location.get(int(class_value)))
        class_names.append(category_file_location.get(class_value))
    
    for k in range(0, num_classes):
        print("Predicted Flower -->", class_names[k])
        print("Probability -->", prob[0,k])
    #print(class_names)
    #print(np.max(probs))
    #z = prob[0,:]
    #y = np.arange(len(class_names))
    #fig=plt.figure(figsize=(5,5))

    #imshow(process_image(image_path))


    #fig= plt.figure(figsize=(10,10))
    #plt.barh( y, z, align='center')
    #plt.yticks(y,class_names)
    #plt.xlabel('Probability')
    #plt.title('Predicted flower')
    #plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',type=str,help="Path of the image location",default='../aipnd-project/flowers')
    parser.add_argument('--checkpoint',type=str,help="path of the checkpoint location",default='./model/checkpoint.pth')
    parser.add_argument('--top_classes',type=int,help="top K predicted classes",default=10)
    parser.add_argument('--category_file_location',type=str,help="location of the json category",default='../aipnd-project/cat_to_name.json')
    parser.add_argument('--device_type',type=str,help="Device used by model",default='gpu')
    args = parser.parse_args()
    #print(args.image_path)
    #print(args.top_classes)
    #print(args.category_file_location)
    #print(args.device_type)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device_type == 'gpu') else "cpu")
    with open(args.category_file_location, 'r') as f:
        cat_to_name = json.load(f)
    
    model,criterion,optimizer,idx_to_class = load_checkpoint(args.checkpoint)
    sanity_check(args.image_path,device,model,cat_to_name,args.top_classes,idx_to_class)
    
if __name__ == '__main__':
    main()