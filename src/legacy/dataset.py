import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WeatherDataset(Dataset):
    def __init__(self, pwv_dir, radar_dir, seq_len_in=10, seq_len_out=20, img_size=(64, 64), sample_step=5):
        self.pwv_dir = pwv_dir
        self.radar_dir = radar_dir
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.total_seq_len = seq_len_in + seq_len_out
        self.sample_step = sample_step
        
        self.pwv_files = []
        self.radar_files = []

        months = sorted([m for m in os.listdir(pwv_dir) if os.path.isdir(os.path.join(pwv_dir, m))])
        
        for month in months:
            month_pwv_dir = os.path.join(pwv_dir, month)
            month_radar_dir = os.path.join(radar_dir, month)
            
            if not os.path.exists(month_radar_dir):
                continue
                
            days = sorted([d for d in os.listdir(month_pwv_dir) if os.path.isdir(os.path.join(month_pwv_dir, d))])
            
            for day in days:
                day_pwv_dir = os.path.join(month_pwv_dir, day)
                day_radar_dir = os.path.join(month_radar_dir, day)
                
                if not os.path.exists(day_radar_dir):
                    continue
                    
                files = sorted([f for f in os.listdir(day_pwv_dir) if f.endswith('.png')])
                
                for f in files:
                    pwv_file_path = os.path.join(day_pwv_dir, f)
                    radar_file_path = os.path.join(day_radar_dir, f)
                    
                    if os.path.exists(radar_file_path) and os.path.exists(pwv_file_path):
                        self.pwv_files.append(pwv_file_path)
                        self.radar_files.append(radar_file_path)

        max_valid_idx = max(0, len(self.pwv_files) - self.total_seq_len)
        self.valid_indices = list(range(0, max_valid_idx + 1, self.sample_step))
        self.n_samples = len(self.valid_indices)
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # 【修复倒置 Bug】：翻转像素值，使得背景为0，风暴为1
            transforms.Lambda(lambda x: 1.0 - x)
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        pwv_seq = []
        radar_seq = []
        
        for i in range(start_idx, start_idx + self.total_seq_len):
            pwv_img = Image.open(self.pwv_files[i])
            radar_img = Image.open(self.radar_files[i])
            
            pwv_seq.append(self.transform(pwv_img))
            radar_seq.append(self.transform(radar_img))
            
        pwv_seq = torch.stack(pwv_seq, dim=0)
        radar_seq = torch.stack(radar_seq, dim=0)
        
        fused_seq = torch.cat([pwv_seq, radar_seq], dim=1)
        
        input_seq = fused_seq[:self.seq_len_in]
        target_seq = fused_seq[self.seq_len_in:]
        
        return input_seq, target_seq
