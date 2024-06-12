import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from custom_dataset import SaliencyDataset # Import the custom dataset
from torchvision import transforms
from model import SimpleCNN  # Import the model from model.py
import matplotlib.pyplot as plt

# Define paths
comic_images_folder = './data/images/'
annotations_folder = './data/maps/'


# Create dataset and split into training and validation sets
train_dataset = SaliencyDataset(comic_images_folder, annotations_folder,phase="train", transform=transform)
val_dataset = SaliencyDataset(comic_images_folder, annotations_folder,phase="val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.MSELoss()  # Assuming regression task for saliency maps
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available and move the model to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to display a batch of images and annotations
def show_batch(sample_batched):
    comic_images_batch, annotations_batch = sample_batched
    batch_size = len(comic_images_batch)
    
    fig, axs = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    for i in range(batch_size):
        axs[i, 0].imshow(comic_images_batch[i].permute(1, 2, 0))
        axs[i, 0].set_title('Comic Image')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(annotations_batch[i].squeeze(), cmap='gray')
        axs[i, 1].set_title('Saliency Annotation')
        axs[i, 1].axis('off')
    
    plt.show()


# Training and validation loops
num_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # Get the inputs
        comic_images, annotations = data
        comic_images, annotations = comic_images.to(device), annotations.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(comic_images)
        loss = criterion(outputs, annotations)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] training loss: {running_loss / 10:.4f}")
            running_loss = 0.0
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            comic_images, annotations = data
            comic_images, annotations = comic_images.to(device), annotations.to(device)
            
            # Forward pass
            outputs = model(comic_images)
            loss = criterion(outputs, annotations)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"[Epoch {epoch + 1}] validation loss: {val_loss:.4f}")

print("Finished Training")