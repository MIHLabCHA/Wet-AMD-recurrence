"""
Author: Hun-gyeom Kim
Date: September, 26, 2025
Email: gnsruatkfkd@gmail.com
Organization: MIHlab
"""

import torch
from torch import nn
from torch.nn import functional as F

# Input resolution used to probe feature dimensions
size = 448


class concat_model(nn.Module):
    # def __init__(self, fe_model2, fe_model4, fe_model6, device, dropout_rate=0.5):
    def __init__(self, fe_model1, fe_model2, fe_model3, fe_model4, fe_model5, fe_model6, device, dropout_rate=0.3):
        super(concat_model, self).__init__()

        # Define feature extractor models and move them to the device
        self.modelA = fe_model1.to(device)
        self.modelB = fe_model2.to(device)
        self.modelC = fe_model3.to(device)
        self.modelD = fe_model4.to(device)
        self.modelE = fe_model5.to(device)
        self.modelF = fe_model6.to(device)

        # Global Average Pooling layer
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Infer output feature dimensions using a dummy input
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, size, size).to(device)
            output_dim_A = self.flatten(self.GAP(self.modelA(sample_input)[-1])).shape[1]
            output_dim_B = self.flatten(self.GAP(self.modelB(sample_input)[-1])).shape[1]
            output_dim_C = self.flatten(self.GAP(self.modelC(sample_input)[-1])).shape[1]
            output_dim_D = self.flatten(self.GAP(self.modelD(sample_input)[-1])).shape[1]
            output_dim_E = self.flatten(self.GAP(self.modelE(sample_input)[-1])).shape[1]
            output_dim_F = self.flatten(self.GAP(self.modelF(sample_input)[-1])).shape[1]

        # Alternative example of inferring dimensions (kept for reference)
        # with torch.no_grad():
        #     sample_input = torch.zeros(1, 3, size, size).to(device)
        #     output_dim_B = self.flatten(self.GAP(self.modelB(sample_input)[-1])).shape[1]
        #     output_dim_D = self.flatten(self.GAP(self.modelD(sample_input)[-1])).shape[1]
        #     output_dim_F = self.flatten(self.GAP(self.modelF(sample_input)[-1])).shape[1]

        # Compute total concatenated feature dimension
        total_output_dim = output_dim_A + output_dim_B + output_dim_C + output_dim_D + output_dim_E + output_dim_F
        # total_output_dim = output_dim_B + output_dim_D + output_dim_F
        print(f"{total_output_dim}")

        # Dropout and fully connected classifier
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(total_output_dim, 128)  # Project concatenated features to 128-D
        self.fc2 = nn.Linear(128, 2)
        self.initialize_weights()

    def forward(self, x1, x2, x3, x4, x5, x6):
    # def forward(self, x2, x4, x6):
        # Get features from each model, apply GAP, then flatten
        x1 = self.flatten(self.GAP(self.modelA(x1)[-1]))
        x2 = self.flatten(self.GAP(self.modelB(x2)[-1]))
        x3 = self.flatten(self.GAP(self.modelC(x3)[-1]))
        x4 = self.flatten(self.GAP(self.modelD(x4)[-1]))
        x5 = self.flatten(self.GAP(self.modelE(x5)[-1]))
        x6 = self.flatten(self.GAP(self.modelF(x6)[-1]))

        # Concatenate all features
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        # x = torch.cat((x2, x4, x6), dim=1)

        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)



class concat_model_1pair_oct_horizon(nn.Module):
    def __init__(self, fe_model3, fe_model4, device, dropout_rate=0.3):
    # def __init__(self, fe_model1, fe_model2, fe_model3, fe_model4, fe_model5, fe_model6, device, dropout_rate=0.3):
        super(concat_model_1pair_oct_horizon, self).__init__()

        # Define feature extractor models and move them to the device
        # self.modelA = fe_model1.to(device)
        # self.modelB = fe_model2.to(device)
        self.modelC = fe_model3.to(device)
        self.modelD = fe_model4.to(device)
        # self.modelE = fe_model5.to(device)
        # self.modelF = fe_model6.to(device)

        # Global Average Pooling layer
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Infer output feature dimensions using a dummy input
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, size, size).to(device)
            # output_dim_A = self.flatten(self.GAP(self.modelA(sample_input)[-1])).shape[1]
            # output_dim_B = self.flatten(self.GAP(self.modelB(sample_input)[-1])).shape[1]
            output_dim_C = self.flatten(self.GAP(self.modelC(sample_input)[-1])).shape[1]
            output_dim_D = self.flatten(self.GAP(self.modelD(sample_input)[-1])).shape[1]
            # output_dim_E = self.flatten(self.GAP(self.modelE(sample_input)[-1])).shape[1]
            # output_dim_F = self.flatten(self.GAP(self.modelF(sample_input)[-1])).shape[1]

        # Compute total concatenated feature dimension
        total_output_dim = output_dim_C + output_dim_D
        # total_output_dim = output_dim_A + output_dim_B + output_dim_C + output_dim_D + output_dim_E + output_dim_F
        # total_output_dim = output_dim_B + output_dim_D + output_dim_F
        print(f"{total_output_dim}")

        # Dropout and fully connected classifier
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(total_output_dim, 128)  # Project concatenated features to 128-D
        self.fc2 = nn.Linear(128, 2)
        self.initialize_weights()

    def forward(self, x3, x4):
    # def forward(self, x2, x4, x6):
        # Get features from each model, apply GAP, then flatten
        # x1 = self.flatten(self.GAP(self.modelA(x1)[-1]))
        # x2 = self.flatten(self.GAP(self.modelB(x2)[-1]))
        x3 = self.flatten(self.GAP(self.modelC(x3)[-1]))
        x4 = self.flatten(self.GAP(self.modelD(x4)[-1]))
        # x5 = self.flatten(self.GAP(self.modelE(x5)[-1]))
        # x6 = self.flatten(self.GAP(self.modelF(x6)[-1]))

        # Concatenate all features
        x = torch.cat((x3, x4), dim=1)
        # x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        # x = torch.cat((x2, x4, x6), dim=1)

        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
