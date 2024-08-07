import torch
from torchvision import models
from torch import nn, optim
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        """
        Initializes a feedforward neural network with customizable hidden layers.

        This constructor builds a neural network with a specified number of hidden layers,
        each with a defined size. It also sets up the output layer and a dropout layer for regularization.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output classes.
            hidden_layers (list of int): A list where each integer represents the number of neurons in
                                        a hidden layer. The list defines the architecture of the hidden layers.
            drop_p (float, optional): Dropout probability for regularization. Default is 0.5.

        Attributes:
            hidden_layers (nn.ModuleList): A list of linear layers representing the hidden layers of the network.
            output (nn.Linear): The final linear layer that transforms the output from the last hidden layer
                                to the output size.
            dropout (nn.Dropout): Dropout layer used to prevent overfitting during training.
        """
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        """
        Forward pass through the neural network.

        Applies a series of hidden layers followed by a ReLU activation function and dropout,
        then passes the result through the output layer. Finally, applies a log softmax activation.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Log probabilities of the output classes.
        """
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        output = F.log_softmax(x, dim=1)
        return output



# Create a classifier base on the user selected model architecture.
def build_model(arch, hidden_layers, learning_rate, train=False):
    """
    Builds and initializes a neural network model based on the specified architecture.

    This function selects a model architecture from a predefined set (ResNet50, VGG16, or DenseNet121),
    replaces the classifier layer with a custom `Classifier` instance, and sets up the optimizer. 
    It also optionally freezes the model's parameters if training is not enabled.

    Args:
        arch (str): The architecture of the model to build. Must be one of 'resnet50', 'vgg16', or 'densenet121'.
        hidden_layers (list of int): A list where each integer specifies the number of neurons in a hidden layer
                                      for the classifier.
        learning_rate (float): The learning rate for the optimizer.
        train (bool, optional): If True, freezes the parameters of the pre-trained model to prevent updates
                                during training. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The initialized neural network model with the custom classifier.
            - optimizer (torch.optim.Optimizer): The Adam optimizer configured for the model's classifier parameters.
    """   

    output_size = 102 # flowers    
    # Set the in_features and base_last_layer base on the arch selected
    if arch == 'resnet50':
        #base_last_layer = model.fc
        model = models.resnet50(weights='DEFAULT' if train else None)
        print(model)
        # Freeze all the parameter so that backpropagation does not affect them        
        if train:
            for param in model.parameters():
                param.requires_grad = False

        in_features = 2048
        classifier = Classifier(input_size=in_features,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)        
    elif arch == 'vgg16':
        #base_last_layer = model.classifier[6]
        model = models.vgg16(weights='DEFAULT' if train else None)
        print(model) 
        # Freeze all the parameter so that backpropagation does not affect them       
        if train:
            for param in model.parameters():
                param.requires_grad = False

        in_features = 4096
        classifier = Classifier(input_size=in_features,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.classifier[6] = classifier
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)
    elif arch == 'densenet121':
        #base_last_layer = model.classifier
        model = models.densenet121(weights='DEFAULT' if train else None)
        print(model)
        # Freeze all the parameter so that backpropagation does not affect them        
        if train:
            for param in model.parameters():
                param.requires_grad = False

        in_features = 1024
        classifier = Classifier(input_size=in_features,
                            output_size=output_size,
                            hidden_layers=hidden_layers,
                            drop_p=0.2)
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return (model, optimizer)

# Function to re-build the model from a checkpoint for inference or more training (train=True).
def load_checkpoint(file_path, train=False):
    """
    Loads a model and optimizer from a checkpoint file.

    This function restores a saved model and optimizer state from a checkpoint file. It reinitializes the model
    and optimizer based on the information stored in the checkpoint, then loads the saved parameters and state.

    Args:
        file_path (str): Path to the checkpoint file.
        train (bool, optional): If True, reinitializes the model to be trainable. If False, the model will be
                                set up as per the checkpoint's saved state. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The restored model with loaded state and classifier.
            - optimizer (torch.optim.Optimizer): The restored optimizer with loaded state.

    Notes:
        The checkpoint file must contain the following keys:
            - 'arch': The architecture of the model used.
            - 'hidden_layers': List of hidden layer sizes for the classifier.
            - 'learning_rate': The learning rate used for the optimizer.
            - 'model_state_dict': The state dictionary of the model.
            - 'class_to_idx': Mapping of class indices to class labels.
            - 'optimizer_state_dict': The state dictionary of the optimizer.
    """    
    checkpoint = torch.load(file_path)    
    model, optimizer = build_model(checkpoint['arch'],
                           checkpoint['hidden_layers'],
                           checkpoint['learning_rate'],
                           train=train)
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (model, optimizer) 

