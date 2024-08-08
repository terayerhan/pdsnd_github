import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)



# Create a classifier base on the user selected model architecture.
def build_model(arch, hidden_layers, learning_rate, train=False):    

    output_size = 102 # flowers    
    # Set the in_features and base_last_layer base on the arch selected
    if arch == 'resnet50':
        #base_last_layer = model.fc
        model = models.resnet50(weights='DEFAULT' if train else None)
        print(model)        
        if train:
            for param in model.parameters():
                param.requires_grad = False
        
        classifier = Classifier(input_size=2048,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)        
    elif arch == 'vgg16':
        #base_last_layer = model.classifier[6]
        model = models.vgg16(weights='DEFAULT' if train else None)
        print(model)        
        if train:
            for param in model.parameters():
                param.requires_grad = False
        
        classifier = Classifier(input_size=4096,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.classifier[6] = classifier
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)
    elif arch == 'densenet121':
        #base_last_layer = model.classifier
        model = models.densenet121(weights='DEFAULT' if train else None)
        print(model)        
        if train:
            for param in model.parameters():
                param.requires_grad = False
        
        classifier = Classifier(input_size=1024,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, optimizer

# Function to re-build the model from a checkpoint for inference or more training (train=True).
def load_checkpoint(file_path, train=False):    
    checkpoint = torch.load(file_path)    
    model, optimizer = build_model(checkpoint['arch'],
                           checkpoint['hidden_layers'],
                           checkpoint['learning_rate'],
                           train=train)
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer 

