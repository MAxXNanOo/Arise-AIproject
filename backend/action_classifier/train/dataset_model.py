import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 1. Custom Dataset สำหรับโหลดไฟล์ .npy มาเทรน
class SPHARGruDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
        
        # ดึงชื่อคลาสทั้งหมดที่มีในชุดข้อมูลมาจัดทำสารบัญคลาส (Unique Classes)
        self.classes = sorted(list(set([os.path.basename(f).split('_')[0] for f in self.file_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # โหลดข้อมูลพิกัด (30, 34)
        sequence = np.load(file_path)
        
        # ดึงป้ายกำกับคลาสจากชื่อไฟล์
        class_name = os.path.basename(file_path).split('_')[0]
        label = self.class_to_idx[class_name]
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 2. โครงสร้างโมเดล GRU สำหรับทำนายท่าทาง
class ActionGRU(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=14):
        super(ActionGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Input x shape: (Batch, Timestep=30, Features=34)
        out, _ = self.gru(x)
        # หยิบเฉพาะความเข้าใจในเฟรมเวลาสุดท้าย (-1) ไปเข้า Fully Connected Classifier
        out = self.fc(out[:, -1, :])
        return out
