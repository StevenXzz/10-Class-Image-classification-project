import os, time
import torch
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from Datasets import Cifar10
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork()
        self.print_model_parameters(self.network)
        self.criterion = nn.CrossEntropyLoss()
        if(self.configs['pretrained']):
            self.pretrained_path = './models/best_model.pth'
            state_dict = torch.load(self.pretrained_path, map_location=self.configs['device'])
            self.network.load_state_dict(state_dict, strict=False)

    def model_setup(self):
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'], momentum=self.configs['momentum'])

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        self.model_setup()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, configs['n_epochs'])
        self.network = self.network.to(configs['device'])
        train_dataset = Cifar10(x_train, y_train, transform=lambda x: parse_record(x, training=True))
        train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
        self.network.train()
        best_acc = 0
        start_epoch = 0
        if self.configs['pretrained']:
            start_epoch = self.configs['start_epoch']
        for epoch in range(start_epoch, configs['n_epochs']):
            start_time = time.time()
            running_loss = 0.0
            total = 0
            correct = 0
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.float() 
                inputs, labels = inputs.to(configs['device']), labels.to(configs['device']).long()
                
                self.optimizer.zero_grad()
                       
                outputs = self.network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
            
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2)
                self.optimizer.step()
                running_loss += loss.item()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                accuracy = correct / total
                avg_loss = running_loss / len(train_loader)
            self.scheduler.step()
            print('Current Learning Rate: {}'.format(self.scheduler.get_last_lr()))
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}')
            if (epoch + 1) % 20 == 0:
                model_path = os.path.join(configs['result_dir'], f'model_epoch_{epoch+1}.pth')
                torch.save(self.network.state_dict(), model_path)
                print(f'Model saved to {model_path}')
            if x_valid is not None and y_valid is not None:
                acc = self.evaluate(x_valid, y_valid)
                if acc > best_acc:
                    best_acc = acc  
                    best_model_path = os.path.join(configs['result_dir'], 'best_model.pth')
                    torch.save(self.network.state_dict(), best_model_path)
                    print(f'New best model with accuracy {best_acc:.4f} saved to {best_model_path}')

        print(f'Best model achieved an accuracy of {best_acc:.4f}')

    def evaluate(self, x, y):
        self.network.eval()  
        self.network = self.network.to(self.configs['device'])
        eval_dataset = Cifar10(x, y, transform=lambda x: parse_record(x, training=False))
        eval_loader = DataLoader(eval_dataset, batch_size=self.configs['batch_size'], shuffle=False)
    
        total_loss = 0
        total_correct = 0
        total = 0
        with torch.no_grad():  
            for inputs, labels in eval_loader:
                inputs = inputs.float().to(self.configs['device'])
                labels = labels.to(self.configs['device']).long()
            
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
            
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                total_loss += loss.item()
    
        avg_loss = total_loss / len(eval_loader)
        accuracy = total_correct / total
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        return accuracy


    def predict_prob(self, x):
        # Ensure the network is in evaluation mode
        self.network.eval()
        self.network = self.network.to(self.configs['device'])  
        # List to hold the processed images
        processed_images = []
    
        # Process each image in the input x
        for i in range(x.shape[0]):
            # Parse and preprocess the record
            image = parse_record(x[i], training=False)
            processed_images.append(image)
    
        # Convert the list of processed images to a numpy array
        processed_images = np.array(processed_images)
    
        # Convert the numpy array to a PyTorch tensor and transfer to the device
        x_tensor = torch.tensor(processed_images, dtype=torch.float32).to(self.configs['device'])
    
        # Predict probabilities with no gradient calculation
        with torch.no_grad():
            output_probs = self.network(x_tensor)
            output_probs = F.softmax(output_probs, dim=1)
        # Transfer the output probabilities to CPU and convert to a numpy array
        probs = output_probs.cpu().numpy()
        return probs
    
    def print_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
