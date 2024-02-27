import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # Output size 16 for 16 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
# Instantiate the model
model = CNNModel()

# Save the model to a .pth file
torch.save(model.state_dict(), 'CNN.pth')

# Load the model from the .pth file
loaded_model = CNNModel()
loaded_model.load_state_dict(torch.load('CNN.pth'))
loaded_model.eval()  # Set the model to evaluation mode if needed
