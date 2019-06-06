import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
import torch.nn.functional as F
import datetime
import os
from Utilities import preprocess_utility
from Utilities.Network import Network

def createModel(model_arch, hidden_layers,learning_rate,output_length,gpu=True):
    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        #print(f" {model_arch} is unsupported model by this application, try vgg16,vgg19 or vgg13")
        raise ValueError(f" {model_arch} is unsupported model by this application, try vgg16,vgg19 or vgg13")
    for param in model.parameters():
        param.requires_grad = False
    
    ###Defining the classifier and tunable params
    vgg_clasifier_input = model.classifier[0].in_features
    
    classifier = Network(input_length=vgg_clasifier_input,
                     output_length=output_length, 
                     hidden_layers=hidden_layers,drop_prob=0.2)
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model,optimizer, criterion

def trainValidate(model,train_dataloader,validate_dataloader,optimizer,criterion,epochs=1,gpu=True):
    print(f"Start of training ----------- {datetime.datetime.now()} ----------- ")
    
    steps = 0
    running_loss = 0
    print_every = 75
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using GPU?", torch.cuda.is_available())
    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validate_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        val_out = model.forward(inputs)
                        batch_loss = criterion(val_out, labels)
                    
                        val_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        val_out_actual = torch.exp(val_out)
                        top_p, top_class = val_out_actual.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {val_loss/len(validate_dataloader):.3f}.. "
                  f"Validation accuracy percent: {accuracy*100/len(validate_dataloader):.3f}")
            
            ##reset running loss, to capture the loss for the duration of next <print_every> steps
                running_loss = 0
                model.train()
    print(f"End of training ------------ {datetime.datetime.now()} ----------- ")        
    return ""
def executeTest(model, test_data, criterion, gpu):
    test_loss = 0
    accuracy = 0
       
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using GPU?", torch.cuda.is_available())
    model.to(device)
    #print(f"The total number of test batch is -> {len(test_data)}")
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        #print(f"The test loss in run {i} is {test_loss}")
        # Convert back to softmax distribution
        ps = torch.exp(output)
        # Compare highest prob predicted class ps.max(dim=1)[1] with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    return test_loss, accuracy*100/(len(test_data))

def saveClassifierModel(model,save_path,arch,optimizer,train_imagefolder,epochs):
    if save_path is None:
        save_path = os.getcwd()
        print(f"No save path specified, saving in current directory {save_path}")
    # Create directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        # Create dictionary of parameters
    param_dict = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_imagefolder.class_to_idx,
        'optimizer': optimizer,
        'epochs': epochs,
        }
    checkpoint_file = "ImageClassifier_"+arch+".pth.tar"
    
    print(param_dict)
    print(f"The file path being saved is {os.path.join(save_path, checkpoint_file)}")
    print("$$$$$$$$$$$$-----Use this checkpoint file for prediction-----$$$$$$$$$$$$")
    torch.save(param_dict, os.path.join(save_path, checkpoint_file))
    return  ""
          
          
def loadClassifierModel(checkpointpath):
    from Utilities.Network import Network
    checkpoint = torch.load(checkpointpath,map_location='cpu')
    ##model = TheModelClass(*args, **kwargs)

    model_onload = models.densenet201(pretrained=True)

    model_onload.classifier = checkpoint['classifier']
    model_onload.load_state_dict = checkpoint['state_dict']
    model_onload.class_to_idx = checkpoint['class_to_idx']
    return model_onload

def predictImageClass(image_path, model, gpu=True,topk=4, category_names=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #test_im = Image.open(image_path)
    image_array = preprocess_utility.predictImagePreprocessor(image_path,256)
    tensor_image = torch.from_numpy(image_array).float()
    ##Adding a new batch dimension to the single image
    tensor_image.unsqueeze_(0)
        
    if gpu:
        # Check GPU availability
        print("Using GPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    tensor_image = tensor_image.to(device)
        
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model.forward(tensor_image)
        ##Get Probabilities
        ps = torch.exp(output)
        probs, indxes = torch.topk(ps, topk)
        ##On CPU for numpy 
        probs = probs.cpu()
        indxes = indxes.cpu()
        ##Converting to numpy aray for easier manipulations
        indxes = indxes.numpy()
        probs=probs.numpy()
               
        probindex = zip(probs[0], indxes[0])
        class_probs=[]
        cell_probs=[]
        print("Show the probability indexes now...")
        print(list(probindex))
        for prob, indx in zip(probs[0], indxes[0]):
            pred_class = [classN for (classN, m_indx) in model.class_to_idx.items() if m_indx == indx]
            print(f"show the class name identified....{pred_class}")
            classNumber = model.class_to_idx[pred_class[0]]
            print(f"show the class number of cellname name identified....{classNumber}")
            cell_probs.append((prob, pred_class[0]))
            # print(classes)
        #print(class_probs)
        print(cell_probs)
    return cell_probs
          
    
    
    

            
            