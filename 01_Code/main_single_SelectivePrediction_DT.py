"""
Code for Wet-AMD recurrence prediction - Selecive prediction (Double threshold)
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
from torch.nn import functional as F
import nni
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import roc_auc_score, roc_curve
import timm
from scipy import stats

from utils.setting import set_seed, create_experiment_path, save_and_load_params

seed = 42
set_seed(seed)

# Image resize size variable (e.g., 448 or 512)
size = 448
print(f"{size}")

# Read Excel file
file_path = '/home/hun-gyeom/WetAMD/Code/Data/all_data/all_data_info.xlsx'
df = pd.read_excel(file_path)

print("Columns in the dataframe:", df.columns)

# Extract required columns
case_numbers = df['Case. No.']
# NOTE: replace the label column name below with your actual column header if different.
labels = df['Recurrence within 12 months after 3 injections (0=no, 1=yes)']

# KFold configuration
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

        # Adjust based on directory layout: expect subfolders class_0 and class_1
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

# Example transforms
default_transform = transforms.Compose([
    transforms.Resize((size, size)),  # use the size variable above
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomSinglePatternDataset(Dataset):
    def __init__(self, img_dir, base_pattern, transforms=None, max_files_per_subject=None):
        """
        img_dir: Root directory containing the images.
        base_pattern: The base pattern for images to include (e.g., 'fun-001').
        transforms: Data transformations to be applied to the images.
        max_files_per_subject: Number of files to sample per subject.
        """
        self.img_dir = img_dir
        self.base_pattern = base_pattern
        self.transforms = transforms
        self.max_files_per_subject = max_files_per_subject
        self.image_paths = []
        self.labels = []

        # Group images by subject
        subject_groups = {}

        for label in [0, 1]:
            class_dir = os.path.join(img_dir, f'class_{label}')
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist.")
                continue
            for img in os.listdir(class_dir):
                # Include images that contain base_pattern or base_pattern_original
                if img.endswith('.png') and (self.base_pattern in img or f"{self.base_pattern}_original" in img):
                    subject_id = self.extract_subject_id(img)
                    if subject_id not in subject_groups:
                        subject_groups[subject_id] = []
                    subject_groups[subject_id].append((os.path.join(class_dir, img), label))

        # Sample up to max_files_per_subject for each subject
        for subject_id, files in subject_groups.items():
            # Always include an 'original' file when present
            original_files = [f for f in files if f"{self.base_pattern}_original" in f[0]]
            if original_files:
                self.image_paths.append(original_files[0][0])
                self.labels.append(original_files[0][1])
                # If 'original' is included, reduce the remaining budget by one
                if self.max_files_per_subject:
                    max_samples = max(0, self.max_files_per_subject - 1)
                    sampled_files = random.sample(
                        [f for f in files if f not in original_files],
                        min(len(files) - 1, max_samples)
                    )
                else:
                    sampled_files = [f for f in files if f not in original_files]
            else:
                sampled_files = random.sample(
                    files,
                    min(len(files), self.max_files_per_subject)
                ) if self.max_files_per_subject else files

            # Add the remaining sampled files
            for file_path, label in sampled_files:
                self.image_paths.append(file_path)
                self.labels.append(label)

    def extract_subject_id(self, img_name):
        # Extract subject ID from filename
        return "-".join(img_name.split('-')[:2])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Use NNI to fetch experiment parameters
params = nni.get_next_parameter()

# Create experiment path and save/load parameters
save_path = create_experiment_path()
trial_params = save_and_load_params(save_path, params)

# Use trial_params afterwards
print(f"Loaded parameters: {trial_params}")
model_name = params.get("FE_model", "efficientnet_b0")  # e.g., efficientnet_b0 / efficientnetv2_rw_s.ra2_in1k / inception_v3
learning_rate = params.get("lr", 0.001)
batch_size = params.get("batchsize", 16)
max_files_per_subject = params.get("max_files_per_subject", 4)

global_lower_bound = params.get("global_lower_bound", 0.4)
global_upper_bound = params.get("global_upper_bound", 0.6)

temperature = params.get("temperature", 1)

# Print each hyperparameter explicitly
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Lower bound: {global_lower_bound}")
print(f"Upper bound: {global_upper_bound}")
print(f"Temperature: {temperature}")

# Build DataLoaders for each fold
fold_dataloaders = []

for fold_number in range(1, 6):
    train_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/train'
    valid_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/val'

    # Use only images whose filenames contain the pattern 'oct-h-002' (e.g., change to 'oct-v-002' if needed)
    train_dataset = CustomSinglePatternDataset(
        img_dir=train_dir,
        base_pattern='oct-h-002',
        transforms=default_transform,
        max_files_per_subject=max_files_per_subject
    )
    valid_dataset = CustomSinglePatternDataset(
        img_dir=valid_dir,
        base_pattern='oct-h-002',
        transforms=default_transform,
        max_files_per_subject=max_files_per_subject
    )
    
    print(f"Fold {fold_number}: Train dataset size = {len(train_dataset)}, Valid dataset size = {len(valid_dataset)}")
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        print(f"Error: Dataset size is 0 in fold {fold_number}. Please check the directory structure and image files.")
        continue
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    fold_dataloaders.append((train_loader, valid_loader))

model = timm.create_model(model_name, pretrained=True, num_classes=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create folder if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

kwargs = {
    'with_NNI': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': model,
    'train_dataset': train_dataset,
    'test_dataset': None,
    'criterion_cls': nn.CrossEntropyLoss(),
    'folds': 5,
    'max_epochs': 70,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'early_stop_threshold': 15,
    'early_stop_criterion': 'auc_score',
    'save_path': save_path,
    'random_state': seed,
    'global_lower_bound': global_lower_bound,
    'global_upper_bound': global_upper_bound,
    'temperature': temperature
}

def apply_temperature_scaling(logits, T=1.0):
    # logits: raw model outputs (N, num_classes), assuming binary classification -> (N, 2)
    # T: temperature scalar
    scaled_logits = logits / T
    scaled_probs = F.softmax(scaled_logits, dim=1)[:, 1]  # probability of positive class
    return scaled_probs

def find_optimal_threshold(labels, probs):
    # Find threshold that maximizes Youden's index
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youdens_index = tpr - fpr
    best_idx = np.argmax(youdens_index)
    best_threshold = thresholds[best_idx]
    return best_threshold

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
    T = kwargs.get('temperature', 2.0)  # temperature parameter
    global_lower_bound = kwargs.get('global_lower_bound', 0.01)
    global_upper_bound = kwargs.get('global_upper_bound', 0.01)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    early_stop_counter = 0

    model.to(device)

    best_auc = 0.0
    best_model_path = None
    train_losses = []
    val_losses = []

    best_metrics = {
        'auc': 0.0,
        'accuracy': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'f1': 0.0,
        'specificity': 0.0,
        'best_threshold': 0.5,
        'confusion_matrix': None,
        'excluded_samples': 0.0
    }

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)  # logits (N, 2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_losses.append(train_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        all_labels = []
        all_probs = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                logits = model(inputs)   # logits (N, 2)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)

                # Apply temperature scaling
                scaled_probs = apply_temperature_scaling(logits, T=T)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(scaled_probs.cpu().numpy())

        val_losses.append(val_loss / len(val_loader.dataset))

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Find optimal threshold via Youden's Index
        best_threshold = find_optimal_threshold(all_labels, all_probs)

        std_value = np.std(all_probs)

        # Apply selective prediction with global bounds
        lower_bound = max(0.0, global_lower_bound)
        upper_bound = min(1.0, global_upper_bound)

        # Selective prediction:
        #   probs < lower_bound  -> Class 0
        #   probs > upper_bound  -> Class 1
        #   otherwise            -> Unknown (-1)
        final_preds = np.full_like(all_labels, fill_value=-1)
        final_preds[all_probs < lower_bound] = 0
        final_preds[all_probs > upper_bound] = 1

        # Accuracy after excluding Unknowns
        mask = (final_preds != -1)
        if np.sum(mask) == 0:
            print(f"Epoch {epoch+1}: No samples left after selective prediction.")
            # Too many Unknowns; continue to next epoch
            continue

        selected_labels = all_labels[mask]
        selected_preds = final_preds[mask]
        selected_probs = all_probs[mask]  # probabilities for selected (non-Unknown) samples

        # Compute AUC (only if both classes are present)
        unique_classes = np.unique(selected_labels)
        if len(unique_classes) == 2:
            auc_score = roc_auc_score(selected_labels, selected_probs)
        else:
            print(f"Epoch {epoch+1}: Only one class present after selective prediction. Setting AUC to 0.")
            auc_score = 0.0

        accuracy = accuracy_score(selected_labels, selected_preds)
        recall = recall_score(selected_labels, selected_preds)
        precision = precision_score(selected_labels, selected_preds)
        f1 = f1_score(selected_labels, selected_preds).item()

        # Confusion matrix and specificity
        cm = confusion_matrix(selected_labels, selected_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Number of Unknown predictions
        excluded_samples = np.mean(final_preds == -1) * len(final_preds)

        # If excluded samples exceed 65 (~80% for a dataset of ~80), set AUC to 0
        if excluded_samples > 65:
            print(f"Epoch {epoch+1}: Excluded samples ({excluded_samples}) exceed 65. Setting AUC to 0.")
            auc_score = 0.0

        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, "
            f"AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, "
            f"Precision: {precision:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}, "
            f"Optimal Threshold: {best_threshold:.4f}, "
            f"Excluded samples: {excluded_samples:.4f}"
        )

        # Early Stopping and model saving
        if auc_score > best_metrics['auc']:
            best_metrics.update({
                'auc': auc_score,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'specificity': specificity,
                'best_threshold': best_threshold,
                'confusion_matrix': confusion_matrix(selected_labels, selected_preds),
                'excluded_samples': excluded_samples
            })

            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'best_model_fold_{fold_number}.pth'))
            print(
                f"Best updated at Epoch {epoch + 1}: AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, "
                f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, "
                f"Specificity: {specificity:.4f}, Best Threshold: {best_threshold:.4f}, "
                f"Excluded samples: {excluded_samples:.4f}"
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

    print(f"Training complete for fold {fold_number}. Best AUC: {best_metrics['auc']:.4f}")

    # Plot Loss Curve
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


def load_weights_for_fold(fold, fe_model):
    """
    Load pre-trained weights for a given fold into fe_model.
    Adjust weight_base_path and model_paths to match your setup.
    """
    weight_base_path = "/data/hun-gyeom/02_WetAMD/Single_result/Fun_OCTh_concat"
    # Examples:
    # /data/hun-gyeom/02_WetAMD/3_weight/SelectivePrediction
    # /data/hun-gyeom/02_WetAMD/Single_result/SelectivePrediction
    
    model_paths = {
        # 'fe_model': f"{weight_base_path}/fun-001/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-h-001/best_model_fold_{fold}.pth"
        'fe_model': f"{weight_base_path}/oct-h-002/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-v-001/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-v-002/best_model_fold_{fold}.pth"
    }

    fe_model.load_state_dict(torch.load(model_paths['fe_model']), strict=True)

    print(f"Loaded weights for fold {fold} from the following paths:")
    for model_name, path in model_paths.items():
        print(f"{model_name}: {path}")

# Train over folds and record metrics
metrics_list = []
for fold, (train_loader, valid_loader) in enumerate(fold_dataloaders, 1):
    print(f"Fold {fold}")

    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    load_weights_for_fold(fold, model)

    kwargs['model'] = model
    kwargs['train_dataset'] = train_loader
    kwargs['test_dataset'] = valid_loader

    best_metrics = train_model(fold_number=fold, **kwargs)
    metrics_list.append(best_metrics)

    print(
        f"Fold {fold} - Best AUC: {best_metrics['auc']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}, "
        f"Recall: {best_metrics['recall']:.4f}, Precision: {best_metrics['precision']:.4f}, F1 Score: {best_metrics['f1']:.4f}, "
        f"Specificity: {best_metrics['specificity']:.4f}, Best Threshold: {best_metrics['best_threshold']:.4f}, "
        f"Excluded samples: {best_metrics['excluded_samples']:.4f}"
    )

# Compute mean and std over 5-fold results
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
    'specificity': np.mean([m['specificity'] for m in metrics_list]),
    'specificity_std': np.std([m['specificity'] for m in metrics_list]),
    'best_threshold': np.mean([m['best_threshold'] for m in metrics_list]),
    'best_threshold_std': np.std([m['best_threshold'] for m in metrics_list]),
    'excluded_samples': np.mean([m['excluded_samples'] for m in metrics_list]),
    'excluded_samples_std': np.std([m['excluded_samples'] for m in metrics_list])
}

# NNI final report
nni.report_final_result(final_metrics['auc'])

# Print mean and std
print(
    f"Final Results: AUC: {final_metrics['auc']:.4f} ± {final_metrics['auc_std']:.4f}, "
    f"Accuracy: {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}, "
    f"Recall: {final_metrics['recall']:.4f} ± {final_metrics['recall_std']:.4f}, "
    f"Precision: {final_metrics['precision']:.4f} ± {final_metrics['precision_std']:.4f}, "
    f"F1 Score: {final_metrics['f1']:.4f} ± {final_metrics['f1_std']:.4f}, "
    f"Specificity: {final_metrics['specificity']:.4f} ± {final_metrics['specificity_std']:.4f}, "
    f"excluded_samples: {final_metrics['excluded_samples']:.4f} ± {final_metrics['excluded_samples_std']:.4f}, "
    f"Best Threshold: {final_metrics['best_threshold']:.4f} ± {final_metrics['best_threshold_std']:.4f}"
)

def save_experiment_details(save_path, model_name, batch_size, learning_rate, final_metrics):
    """Save experiment details to a text file."""
    details_path = os.path.join(save_path, 'experiment_details.txt')
    with open(details_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"global_lower_bound: {global_lower_bound}\n")
        f.write(f"global_upper_bound: {global_upper_bound}\n")
        f.write("\nFinal Metrics:\n")
        f.write(f"AUC: {final_metrics['auc']:.4f} ± {final_metrics['auc_std']:.4f}\n")
        f.write(f"Accuracy: {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}\n")
        f.write(f"Recall: {final_metrics['recall']:.4f} ± {final_metrics['recall_std']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f} ± {final_metrics['precision_std']:.4f}\n")
        f.write(f"F1 Score: {final_metrics['f1']:.4f} ± {final_metrics['f1_std']:.4f}\n")
        f.write(f"Specificity: {final_metrics['specificity']:.4f} ± {final_metrics['specificity_std']:.4f}\n")
        f.write(f"excluded_samples: {final_metrics['excluded_samples']:.4f} ± {final_metrics['excluded_samples_std']:.4f}\n")
        f.write(f"Best Threshold: {final_metrics['best_threshold']:.4f} ± {final_metrics['best_threshold_std']:.4f}\n")

# Save experiment details to a text file
save_experiment_details(save_path, model_name, batch_size, learning_rate, final_metrics)

# Save all metrics to output.pkl
with open(os.path.join(save_path, 'output.pkl'), 'wb') as f:
    pickle.dump(metrics_list, f)
