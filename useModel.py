import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

# Define the CNN model for bounding box regression
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # Output 4 values for bounding box coordinates

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)  # Output 4 values for bounding box coordinates
        return x

# Load the trained model
net = CNN()
net.load_state_dict(torch.load('trained_model.pth'))
net.eval()  # Set the model to evaluation mode

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and preprocess your image
image = Image.open('/home/itpat/Code/COSI/Drone/Crowd_Localization/ProcessedData/JHU/images/4365.jpg')
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = net(input_batch)

# Process the output to interpret predictions
predicted_bbox = output[0]  # Assuming output is a tensor of shape (1, 4) for the bounding box coordinates

# Display the image with predicted bounding box
plt.imshow(image)
plt.title('Predicted Bounding Box')
plt.axis('off')

# Convert predicted bounding box coordinates to (x, y, width, height) format
x, y, w, h = predicted_bbox.tolist()

# Scale the coordinates to match the resized image
x *= image.width / 32
y *= image.height / 32
w *= image.width / 32
h *= image.height / 32

# Create a Rectangle patch with color
rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')  # Red bounding box outline
# Alternatively, you can use 'blue' for the edge color and 'cyan' for the face color

# Add the patch to the Axes
plt.gca().add_patch(rect)

plt.show()
