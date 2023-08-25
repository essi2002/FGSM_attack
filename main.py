import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from train import train
from mnist_dataset import mnist_dataset
import torch.nn as nn

def fgsm_attack(model, loss, images, labels, eps) :
    
    images = images.to(torch.float32) 
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, torch.argmax(labels,dim = 1))
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def imshow(images, labels,path):
    image_array = images[0].detach().numpy()  # Detach the tensor from the computational graph
    label = labels[0]

    # Create a plot to display the image
    plt.imshow(image_array.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    # Save the image
    save_path = f"./{path}.png"
    plt.imsave(save_path, image_array.squeeze(), cmap='gray',dpi = 300)
    print(f"Image saved at: {save_path}")
def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(root='./datasets/', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='./datasets/', train=True, download=False, transform=transform)
    train_loader = DataLoader(dataset=mnist_dataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=mnist_dataset(test_data), batch_size=64, shuffle=True)
   
    images, labels = next(iter(train_loader))
   
    imshow(images, labels,"normal")
    print(train_loader.dataset.data.shape)
    model = train(train_loader)
    perturb_image = fgsm_attack(model,nn.CrossEntropyLoss(),images,labels,0.007)
    imshow(perturb_image,labels,"adversial")
if __name__ == "__main__":
    main()