"""
Code for Wet-AMD recurrence prediction
Author: Hun-gyeom Kim
Date: September, 26, 2025
Email: gnsruatkfkd@gmail.com
Organization: MIHlab

If you use this code, please cite:
"Uncertainty-Aware Selective Prediction of Neovascular Age-Related Macular Degeneration Recurrence Using Artificial Intelligence"
"""

import pandas as pd
import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from collections import OrderedDict
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
import torch.optim as optim
import nni
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import timm

from utils.setting import set_seed, create_experiment_path, save_and_load_params

seed = 42
set_seed(seed)

# Image resize (try 448 or 512)
size = 448
print(f"{size}")

# Read Excel file
file_path = '/home/hun-gyeom/WetAMD/Code/Data/all_data/all_data_info.xlsx'
# file_path = '/home/hun-gyeom/WetAMD/Code/Data/all_data/all_data_info_medi_R1.xlsx'
# file_path = '/home/hun-gyeom/WetAMD/Code/Data/all_data/all_data_info_medi_A2.xlsx'

df = pd.read_excel(file_path)
print("Columns in the dataframe:", df.columns)

# (Optional) Example: extract specific columns if needed for analysis
# case_numbers = df['Case. No.']           # e.g., patient case number
# labels = df['recurrence_12mo']           # e.g., 0 = no recurrence, 1 = recurrence

# Stratified KFold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

class CustomDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        """
        img_dir: Root directory containing the images.
        transforms: Data transformations to be applied to the images.
        """
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_paths = []
        self.labels = []

        # Adjust the directory structure based on the provided folder layout
        for label in [0, 1]:
            class_dir = os.path.join(img_dir, f'class_{label}')
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist.")
                continue
            for img in os.listdir(class_dir):
                if img.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Define transforms
default_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomSinglePatternDataset(Dataset):
    def __init__(self, img_dir, base_pattern, transforms=None, max_files_per_subject=None):
        self.img_dir = img_dir
        self.base_pattern = base_pattern
        self.transforms = transforms
        self.max_files_per_subject = max_files_per_subject
        self.image_paths = []
        self.labels = []

        subject_groups = {}

        for label in [0, 1]:
            class_dir = os.path.join(img_dir, f'class_{label}')
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist.")
                continue
            for img in os.listdir(class_dir):
                if img.endswith('.png') and (self.base_pattern in img or f"{self.base_pattern}_original" in img):
                    subject_id = self.extract_subject_id(img)
                    if subject_id not in subject_groups:
                        subject_groups[subject_id] = []
                    subject_groups[subject_id].append((os.path.join(class_dir, img), label))

        for subject_id, files in subject_groups.items():
            original_files = [f for f in files if f"{self.base_pattern}_original" in f[0]]
            if original_files:
                self.image_paths.append(original_files[0][0])
                self.labels.append(original_files[0][1])
                if self.max_files_per_subject:
                    max_samples = max(0, self.max_files_per_subject - 1)
                    sampled_files = random.sample([f for f in files if f not in original_files], min(len(files) - 1, max_samples))
                else:
                    sampled_files = [f for f in files if f not in original_files]
            else:
                sampled_files = random.sample(files, min(len(files), self.max_files_per_subject)) if self.max_files_per_subject else files
            
            for file_path, label in sampled_files:
                self.image_paths.append(file_path)
                self.labels.append(label)

        # Debug (optional): verify image paths and labels after dataset initialization
        # print("Dataset Initialization Complete. Verifying image paths and labels:")
        # for i in range(len(self.image_paths)):
        #     print(f"Path: {self.image_paths[i]}, Label: {self.labels[i]}")

    def extract_subject_id(self, img_name):
        return "-".join(img_name.split('-')[:2])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise e

        if self.transforms:
            try:
                image = self.transforms(image)
            except Exception as e:
                raise e

        return image, torch.tensor(label, dtype=torch.float32), img_path  

# Get hyperparameters from NNI
params = nni.get_next_parameter()

# Create experiment path and save/load params
save_path = create_experiment_path()
trial_params = save_and_load_params(save_path, params)

print(f"Loaded parameters: {trial_params}")
model_name = params.get("FE_model", "resnet50")
learning_rate = params.get("lr", 0.001)
batch_size = params.get("batchsize", 32)
max_files_per_subject = params.get("max_files_per_subject", 4)  # Added: max_files_per_subject

print(f"Model_name: {model_name}")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")

# Create DataLoaders for each fold
fold_dataloaders = []

for fold_number in range(1, 6):
    train_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/train'  # augmented_good / augmented_good_R1 / augmented_good_A2
    valid_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/val'

    # Use only images whose filename contains the given base pattern (e.g., 'fun-001', 'fun-002', 'oct-h-001', 'oct-v-002')
    base_pattern = 'fun-002'

    train_dataset = CustomSinglePatternDataset(img_dir=train_dir, base_pattern=base_pattern, transforms=default_transform, max_files_per_subject=max_files_per_subject)
    valid_dataset = CustomSinglePatternDataset(img_dir=valid_dir, base_pattern=base_pattern, transforms=default_transform, max_files_per_subject=max_files_per_subject)
    
    print(f"Fold {fold_number}: Train dataset size = {len(train_dataset)}, Valid dataset size = {len(valid_dataset)}")
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        print(f"Error: Dataset size is 0 in fold {fold_number}. Please check the directory structure and image files.")
        continue
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    fold_dataloaders.append((train_loader, valid_loader))

# resnet50 / inception_v3 / resnet152 / densenet121 / efficientnet_b0 / vgg16
print("Starting model initialization...")
model = timm.create_model(model_name, pretrained=True, num_classes=2)
print("Model initialization complete")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"{model_name}")

# Build per-experiment folder path
for_save_path = '/data/hun-gyeom/02_WetAMD/Experiment_Log/'
experiment_id = nni.get_experiment_id()
trial_id = nni.get_trial_id()

experiment_path = os.path.join(for_save_path, experiment_id)
save_path = os.path.join(experiment_path, trial_id)

# Create folder if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

kwargs = {
    'with_NNI': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': model,
    'train_dataset': None,
    'test_dataset': None,
    'criterion_cls': nn.CrossEntropyLoss(),
    'folds': 5,
    'max_epochs': 5,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'early_stop_threshold': 25,
    'early_stop_criterion': 'auc_score',
    'save_path': save_path,
    'random_state': seed
}

def train_model(fold_number, **kwargs):
    device = kwargs.get('device', 'cpu')
    model = kwargs.get('model')
    train_loader = kwargs.get('train_dataset')
    val_loader = kwargs.get('test_dataset')
    criterion = kwargs.get('criterion_cls')
    max_epochs = kwargs.get('max_epochs', 100)
    batch_size = kwargs.get('batch_size', 2)
    learning_rate = kwargs.get('learning_rate', 0.001)
    early_stop_threshold = kwargs.get('early_stop_threshold', 10)
    early_stop_criterion = kwargs.get('early_stop_criterion', 'auc_score')
    save_path = kwargs.get('save_path')

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_threshold = 0.5
    early_stop_counter = 0

    model.to(device)

    train_losses = []
    val_losses = []
    auc_scores = []

    best_metrics = {
        'auc': 0.0,
        'accuracy': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'f1': 0.0,
        'best_threshold': 0.5,
        'confusion_matrix': None
    }

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels, paths) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Debug (optional): verify that image paths match the labels
            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}:")
            # print(f"Inputs (shape): {inputs.shape}")
            # print(f"Labels: {labels}")
            # for i in range(len(labels)):
            #     print(f"  Path: {paths[i]}, Label: {labels[i].item()}")

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels, paths in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                
                probabilities = F.softmax(outputs, dim=1)
                probabilities = probabilities[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(probabilities.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs).flatten()

        fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
        youdens_index = tpr - fpr
        best_threshold_index = np.argmax(youdens_index)
        best_threshold = thresholds[best_threshold_index]

        auc_score = roc_auc_score(all_labels, all_outputs)
        auc_scores.append(auc_score)

        pred_labels = (all_outputs > best_threshold).astype(int)

        accuracy = accuracy_score(all_labels, pred_labels)
        recall = recall_score(all_labels, pred_labels)
        precision = precision_score(all_labels, pred_labels)
        f1 = f1_score(all_labels, pred_labels)
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")

        if auc_score > best_metrics['auc']:
            best_metrics.update({
                'auc': auc_score,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'best_threshold': best_threshold,
                'confusion_matrix': confusion_matrix(all_labels, pred_labels)
            })

            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'best_model_fold_{fold_number}.pth'))
            print(f"Best updated at Epoch {epoch+1}: AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}", f"Best Threshold: {best_threshold:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()

    print(f"Training complete for fold {fold_number}. Best AUC: {best_metrics['auc']:.4f}")

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 5)
    plt.title(f'Fold {fold_number} Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'loss_plot_fold_{fold_number}.png'))
    plt.close()

    if best_metrics['confusion_matrix'] is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=best_metrics['confusion_matrix'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Fold {fold_number} Confusion Matrix')
        plt.savefig(os.path.join(save_path, f'confusion_matrix_fold_{fold_number}.png'))
        plt.close()

    nni.report_intermediate_result(best_metrics['auc'])

    return best_metrics

# Train per fold and collect metrics
metrics_list = []
for fold, (train_loader, valid_loader) in enumerate(fold_dataloaders, 1):
    print(f"Fold {fold}")

    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    kwargs['model'] = model
    kwargs['train_dataset'] = train_loader
    kwargs['test_dataset'] = valid_loader

    best_metrics = train_model(fold_number=fold, **kwargs)
    metrics_list.append(best_metrics)

    print(f"Fold {fold} - Best AUC: {best_metrics['auc']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}, "
          f"Recall: {best_metrics['recall']:.4f}, Precision: {best_metrics['precision']:.4f}, F1 Score: {best_metrics['f1']:.4f}, "
          f"Best Threshold: {best_metrics['best_threshold']:.4f}")

# Compute mean and std across folds
final_metrics = {
    'auc': np.mean([m['auc'] for m in metrics_list]),
    'auc_std': np.std([m['auc'] for m in metrics_list]),
    'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
    'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
    'recall': np.mean([m['recall'] for m in metrics_list]),
    'recall_std': np.std([m['recall'] for m in metrics_list]),
    'precision': np.mean([m['precision'] for m in metrics_list]),
    'precision_std': np.std([m['precision'] for m in metrics_list]),
    'f1': np.mean([m['f1'] for m in metrics_list]),
    'f1_std': np.std([m['f1'] for m in metrics_list]),
    'best_threshold': np.mean([m['best_threshold'] for m in metrics_list]),
    'best_threshold_std': np.std([m['best_threshold'] for m in metrics_list])
}

# NNI final report
nni.report_final_result(final_metrics['auc'])

# Print mean and std results
print(f"Final Results: AUC: {final_metrics['auc']:.4f} ± {final_metrics['auc_std']:.4f}, "
      f"Accuracy: {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}, "
      f"Recall: {final_metrics['recall']:.4f} ± {final_metrics['recall_std']:.4f}, "
      f"Precision: {final_metrics['precision']:.4f} ± {final_metrics['precision_std']:.4f}, "
      f"F1 Score: {final_metrics['f1']:.4f} ± {final_metrics['f1_std']:.4f}, "
      f"Best Threshold: {final_metrics['best_threshold']:.4f} ± {final_metrics['best_threshold_std']:.4f}")

def save_experiment_details(save_path, model_name, batch_size, learning_rate, final_metrics):
    """Save experiment details to a text file."""
    details_path = os.path.join(save_path, 'experiment_details.txt')
    with open(details_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write("\nFinal Metrics:\n")
        f.write(f"AUC: {final_metrics['auc']:.4f} ± {final_metrics['auc_std']:.4f}\n")
        f.write(f"Accuracy: {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}\n")
        f.write(f"Recall: {final_metrics['recall']:.4f} ± {final_metrics['recall_std']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f} ± {final_metrics['precision_std']:.4f}\n")
        f.write(f"F1 Score: {final_metrics['f1']:.4f} ± {final_metrics['f1_std']:.4f}\n")
        f.write(f"Best Threshold: {final_metrics['best_threshold']:.4f} ± {final_metrics['best_threshold_std']:.4f}\n")

# Save experiment details to a text file
save_experiment_details(save_path, model_name, batch_size, learning_rate, final_metrics)

# Save all metrics to output.pkl
with open(os.path.join(save_path, 'output.pkl'), 'wb') as f:
    pickle.dump(metrics_list, f)
