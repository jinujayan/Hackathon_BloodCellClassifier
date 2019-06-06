import torch
from torchvision import datasets, transforms,models
import json
from PIL import Image
import numpy as np

###Transformation constants
dataset_mean=[0.485, 0.456, 0.406]
dataset_std=[0.229, 0.224, 0.225]
testdata_rotate= 30
testdata_v_flip=0.5
testdata_h_flip=0.5
resize_crop = 224


def loadTransformData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    

    train_transforms = transforms.Compose([transforms.RandomRotation(testdata_rotate),
                                       transforms.RandomResizedCrop(resize_crop),
                                       transforms.RandomHorizontalFlip(testdata_h_flip),
                                       transforms.RandomVerticalFlip(testdata_v_flip),
                                       transforms.ToTensor(),
                                       transforms.Normalize(dataset_mean,
                                                            dataset_std)])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(resize_crop),
                                            transforms.ToTensor(),
                                           transforms.Normalize(dataset_mean,
                                                            dataset_std)])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(resize_crop),
                                            transforms.ToTensor(),
                                           transforms.Normalize(dataset_mean,
                                                            dataset_std)])


# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the transforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, batch_size=32,shuffle=True)

    return train_dataloader,validate_dataloader,test_dataloader,train_data

###Function to read the json look up file
def loadJSONClassLookup(jsonfile = "cat_to_name.json"):
    with open(jsonfile, 'r') as f:
        classNumber_to_name = json.load(f)
        keyset = set(classNumber_to_name.keys())
        class_count=len(keyset)
    return classNumber_to_name, class_count

def predictImagePreprocessor(image_path,lowside):
    test_im = Image.open(image_path)
    width, height = test_im.size
    #print("The image size is W-> {width} H-> {height}")

    
    ##For a constant aspect ratio find the side given one side
    k = width/height
    if width< height:
        new_w = lowside
        new_h = int(lowside/k)
    else:
        new_w=int(lowside*k)
        new_h=lowside
    new_size=(new_w,new_h)
    test_im.thumbnail(new_size)
    
    crop_new_w,crop_new_h=(224,224)
    left = int((new_w - crop_new_w)/2)
    top = int((new_h - crop_new_h)/2)
    right=left+crop_new_w
    bottom=top+crop_new_w
      
    crpd_image = test_im.crop((left, top, right, bottom))
    np_image = np.array(crpd_image)
    
    ##Normalize(min-max scaling) grayscale values from 0-255 to 0-1
    np_image = np_image/255
    
    ##Standardize(Z-score normalization) of each pixel value using a mean and std
    Z_im = (np_image - dataset_mean)/dataset_std
    print(f"Shape of transposed np_image is {Z_im.transpose().shape}")
    Z_im = Z_im.transpose()
    return Z_im
    



