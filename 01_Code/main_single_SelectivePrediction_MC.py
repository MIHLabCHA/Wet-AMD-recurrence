"""
Code for Wet-AMD recurrence prediction - Selecive prediction (MC dropout)
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
import timm
from scipy import stats

from utils.setting import set_seed, create_experiment_path, save_and_load_params

seed = 42
set_seed(seed)

# Image resize size (e.g., 448 or 512)
size = 448
print(f"{size}")

# Read metadata Excel file (adjust path and column headers as needed)
file_path = '/home/hun-gyeom/WetAMD/Code/Data/all_data/all_data_info.xlsx'
df = pd.read_excel(file_path)
print("Columns in the dataframe:", df.columns)

# If you need specific columns, adapt the names below to your sheet.
# These variables are not used by the training loop; keep or remove as you prefer.
case_numbers = df.get('Case. No.', None)
# Example (uncomment and rename to match your sheet if you plan to use it):
# labels = df.get('Recurrence (12 months after 3 injections) [0=No, 1=Yes]', None)

# (Optional) KFold config if needed elsewhere
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

        # Adjust to directory layout: images under {img_dir}/class_0 and {img_dir}/class_1
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

# Default transforms example
default_transform = transforms.Compose([
    transforms.Resize((size, size)),  # use the variable 'size'
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomSinglePatternDataset(Dataset):
    def __init__(self, img_dir, base_pattern, transforms=None, max_files_per_subject=None):
        """
        img_dir: Root directory containing the images.
        base_pattern: The filename substring to include (e.g., 'fun-001').
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
                # Include only files matching base_pattern or base_pattern + '_original'
                if img.endswith('.png') and (self.base_pattern in img or f"{self.base_pattern}_original" in img):
                    subject_id = self.extract_subject_id(img)
                    if subject_id not in subject_groups:
                        subject_groups[subject_id] = []
                    subject_groups[subject_id].append((os.path.join(class_dir, img), label))

        # Sample up to 'max_files_per_subject' from each subject (always keep the '_original' if present)
        for subject_id, files in subject_groups.items():
            original_files = [f for f in files if f"{self.base_pattern}_original" in f[0]]
            if original_files:
                # Always include one original
                self.image_paths.append(original_files[0][0])
                self.labels.append(original_files[0][1])
                # Reduce remaining budget by one (for the original)
                if self.max_files_per_subject:
                    max_samples = max(0, self.max_files_per_subject - 1)
                    sampled_files = random.sample(
                        [f for f in files if f not in original_files],
                        min(len(files) - 1, max_samples)
                    )
                else:
                    sampled_files = [f for f in files if f not in original_files]
            else:
                sampled_files = (
                    random.sample(files, min(len(files), self.max_files_per_subject))
                    if self.max_files_per_subject else files
                )

            # Add the rest
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

# Fetch experiment parameters from NNI
params = nni.get_next_parameter()

# Create experiment path, then save & reload parameters
save_path = create_experiment_path()
trial_params = save_and_load_params(save_path, params)

# Use trial_params thereafter
print(f"Loaded parameters: {trial_params}")
model_name = params.get("FE_model", "efficientnet_b0")
learning_rate = params.get("lr", 0.001)
batch_size = params.get("batchsize", 8)
max_files_per_subject = params.get("max_files_per_subject", 4)
K_value = params.get("Kvalue", 1.28)

# Log hyperparameters
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"K Value: {K_value}")

# Build DataLoaders for each fold
fold_dataloaders = []
for fold_number in range(1, 6):
    # Options: augmented_good, augmented_good_R1, augmented_good_A2
    train_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/train'
    valid_dir = f'/data/hun-gyeom/02_WetAMD/2_Data/augmented_good/fold_{fold_number}/val'

    # Train/validate using images matching a specific pattern (e.g., 'oct-h-002'; change if needed)
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

# Ensure output directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

kwargs = {
    'with_NNI': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': None,                 # will be set per fold
    'train_dataset': None,         # will be set per fold (DataLoader)
    'test_dataset': None,          # will be set per fold (DataLoader)
    'criterion_cls': nn.CrossEntropyLoss(),
    'folds': 5,
    'max_epochs': 70,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'early_stop_threshold': 25,
    'early_stop_criterion': 'auc_score',
    'save_path': save_path,
    'random_state': seed,
    'K_value': K_value
}

def mc_dropout_predict(model, inputs, num_samples=30, base_seed=42):
    """
    Perform MC Dropout by enabling dropout (and droppath) during inference,
    while keeping BatchNorm layers in eval mode (fixed statistics).
    Returns mean and variance of softmax probabilities over 'num_samples' runs.
    """
    predictions = []

    with torch.no_grad():
        # Set entire model to train mode so dropout is active, but freeze BatchNorm layers
        model.train()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.eval()

        for i in range(num_samples):
            # Different seed each iteration to maintain dropout randomness
            torch.manual_seed(base_seed + i*3)
            np.random.seed(base_seed + i*3)
            random.seed(base_seed + i*3)
            torch.cuda.manual_seed(base_seed + i*3)

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions.append(probabilities.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)
    return torch.mean(predictions, dim=0).cpu().numpy(), torch.var(predictions, dim=0).cpu().numpy()

def select_by_uncertainty(all_uncertainties, K_value=1.0, method='std', q=90):
    """
    Returns:
        mask: boolean array (True = keep)
        threshold: the numeric threshold used
    Methods:
      - 'std'        : mean + K * std               (default; recommended)
      - 'percentile' : keep bottom q% (e.g., q=90)
      - 'sem'        : mean + K * std/sqrt(N)      
    """
    mu = np.mean(all_uncertainties)
    sigma = np.std(all_uncertainties)
    N = len(all_uncertainties)

    if method == 'std':
        thr = mu + K_value * sigma
    elif method == 'percentile':
        thr = np.percentile(all_uncertainties, q)
    elif method == 'sem':  
        thr = mu + K_value * (sigma / np.sqrt(N))
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    mask = all_uncertainties < thr
    return mask, thr


def train_model(fold_number, **kwargs):
    device = kwargs.get('device', 'cpu')
    model = kwargs.get('model')
    train_loader = kwargs.get('train_dataset')
    val_loader = kwargs.get('test_dataset')
    criterion = kwargs.get('criterion_cls')
    max_epochs = kwargs.get('max_epochs', 100)
    learning_rate = kwargs.get('learning_rate', 0.001)
    early_stop_threshold = kwargs.get('early_stop_threshold', 10)
    early_stop_criterion = kwargs.get('early_stop_criterion', 'auc_score')
    save_path = kwargs.get('save_path')
    K_value = kwargs.get('K_value')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_losses.append(train_loss / len(train_loader.dataset))

        # Validation with uncertainty estimation (MC Dropout)
        model.eval()
        all_labels = []
        all_outputs = []
        all_uncertainties = []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs_mean, outputs_var = mc_dropout_predict(model, inputs, num_samples=30)
                probabilities = outputs_mean[:, 1]  # probability for class 1

                # Note: criterion expects logits; here we pass probabilities for logging only.
                val_loss += criterion(torch.tensor(outputs_mean).to(device), labels).item() * inputs.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(probabilities)
                all_uncertainties.extend(outputs_var[:, 1])

        val_losses.append(val_loss / len(val_loader.dataset))

        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs).flatten()
        all_uncertainties = np.array(all_uncertainties).flatten()

        # Compute uncertainty threshold
        mean_uncertainty = np.mean(all_uncertainties)
        std_uncertainty = np.std(all_uncertainties)
        
        uncertainty_method     = kwargs.get('uncertainty_method', 'std')      # 'std' | 'percentile' | 'sem'
        uncertainty_percentile = kwargs.get('uncertainty_percentile', 90)     # used only if method == 'percentile'

        selected_indices, uncertainty_threshold = select_by_uncertainty(
            all_uncertainties,
            K_value=K_value,
            method=uncertainty_method,
            q=uncertainty_percentile
        )

        excluded_samples = len(all_uncertainties) - np.sum(selected_indices)
        selected_labels = all_labels[selected_indices]
        selected_outputs = all_outputs[selected_indices]

        # If only one class remains after filtering, set metrics to zero
        if len(np.unique(selected_labels)) < 2:
            print(f"Epoch {epoch + 1}: Only one class in selected_labels. Metrics set to 0.")
            auc_score = 0
            accuracy = 0
            recall = 0
            precision = 0
            f1 = 0
            specificity = 0
        else:
            auc_score = roc_auc_score(selected_labels, selected_outputs)
            fpr, tpr, thresholds = roc_curve(selected_labels, selected_outputs)
            youdens_index = tpr - fpr
            best_threshold_index = np.argmax(youdens_index)
            best_threshold = thresholds[best_threshold_index]

            pred_labels = (selected_outputs > best_threshold).astype(int)
            accuracy = accuracy_score(selected_labels, pred_labels)
            recall = recall_score(selected_labels, pred_labels)
            precision = precision_score(selected_labels, pred_labels)
            f1 = f1_score(selected_labels, pred_labels)
            tn, fp, fn, tp = confusion_matrix(selected_labels, pred_labels).ravel()
            specificity = tn / (tn + fp)

        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
            f"AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, "
            f"Precision: {precision:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}, "
            f"Uncertainty Mean: {np.mean(all_uncertainties):.4f}, Uncertainty Std: {np.std(all_uncertainties):.4f}, "
            f"Excluded samples: {excluded_samples:.4f}"
        )

        # Save best model (by AUC)
        if auc_score > best_metrics['auc']:
            best_metrics.update({
                'auc': auc_score,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'specificity': specificity,
                'best_threshold': best_threshold,
                'confusion_matrix': confusion_matrix(selected_labels, pred_labels),
                'excluded_samples': excluded_samples
            })

            # Plot and save uncertainty distribution histogram
            plt.figure()
            plt.hist(all_uncertainties, bins=30, density=True, alpha=0.6)
            plt.xlabel('Uncertainty')
            plt.ylabel('Density')
            plt.title('Uncertainty Distribution')
            plt.savefig(os.path.join(save_path, f'fold {fold_number}_uncertainty_distribution_epoch_{epoch + 1}.png'))
            plt.close()

            # Normality tests (Shapiro-Wilk and Kolmogorov-Smirnov)
            shapiro_test = stats.shapiro(all_uncertainties)
            ks_test = stats.kstest(all_uncertainties, 'norm', args=(np.mean(all_uncertainties), np.std(all_uncertainties)))
            # (Optional) Inspect results if needed:
            # print("Shapiro:", shapiro_test, "KS:", ks_test)

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
        disp.plot()
        plt.title(f'Fold {fold_number} Confusion Matrix')
        plt.savefig(os.path.join(save_path, f'confusion_matrix_fold_{fold_number}.png'))
        plt.close()

    nni.report_intermediate_result(best_metrics['auc'])

    return best_metrics

def load_weights_for_fold(fold, fe_model):
    """
    Load pre-trained weights for the given fold into 'fe_model'.
    Adjust 'weight_base_path' and subfolders as needed.
    """
    weight_base_path = "/data/hun-gyeom/02_WetAMD/Single_result/Fun_OCTh_concat"
    # Other examples:
    # /data/hun-gyeom/02_WetAMD/3_weight/SelectivePrediction
    # /data/hun-gyeom/02_WetAMD/Single_result/SelectivePrediction
    # /data/hun-gyeom/02_WetAMD/Single_result/20250314_additional_medicine/Only_A2_normal

    model_paths = {
        # 'fe_model': f"{weight_base_path}/fun-001/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/fun-002/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-h-001/best_model_fold_{fold}.pth"
        'fe_model': f"{weight_base_path}/oct-h-002/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-v-001/best_model_fold_{fold}.pth"
        # 'fe_model': f"{weight_base_path}/oct-v-002/best_model_fold_{fold}.pth"
    }

    # Load weights (strict=True for exact match)
    fe_model.load_state_dict(torch.load(model_paths['fe_model']), strict=True)

    print(f"Loaded weights for fold {fold} from the following paths:")
    for model_name, path in model_paths.items():
        print(f"{model_name}: {path}")

# Train for each fold and collect metrics
metrics_list = []
for fold, (train_loader, valid_loader) in enumerate(fold_dataloaders, 1):
    print(f"Fold {fold}")

    model = timm.create_model(model_name, pretrained=True, num_classes=2, drop_rate=0.1)
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
        f"Recall: {best_metrics['recall']:.4f}, Precision: {best_metrics['precision']:.4f}, "
        f"F1 Score: {best_metrics['f1']:.4f}, Specificity: {best_metrics['specificity']:.4f}, "
        f"Best Threshold: {best_metrics['best_threshold']:.4f}, Excluded samples: {best_metrics['excluded_samples']:.4f}"
    )

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
    'specificity': np.mean([m['specificity'] for m in metrics_list]),
    'specificity_std': np.std([m['specificity'] for m in metrics_list]),
    'best_threshold': np.mean([m['best_threshold'] for m in metrics_list]),
    'best_threshold_std': np.std([m['best_threshold'] for m in metrics_list]),
    'excluded_samples': np.mean([m['excluded_samples'] for m in metrics_list]),
    'excluded_samples_std': np.std([m['excluded_samples'] for m in metrics_list])
}

# NNI final report
nni.report_final_result(final_metrics['auc'])

# Print summary
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
    """Save key experiment details to a text file."""
    details_path = os.path.join(save_path, 'experiment_details.txt')
    with open(details_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"K_value: {K_value}\n")
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

# Save all per-fold metrics to output.pkl
with open(os.path.join(save_path, 'output.pkl'), 'wb') as f:
    pickle.dump(metrics_list, f)
