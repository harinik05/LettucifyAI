import Model as CNNModel

model = CNNModel()
model.load_state_dict(torch.load("CNN_Model.pth"))
model.eval()