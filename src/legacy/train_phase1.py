import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataset import WeatherDataset
from torch.utils.data import DataLoader, Subset
from model import WeatherPredictor

# ==========================================
# 1. 空间与时间约束工具
# ==========================================
def create_spatial_mask(H, W, device, sigma=15.0):
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device).float() - cy
    x = torch.arange(W, device=device).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    mask = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    mask = mask / mask.max()
    return mask.view(1, 1, 1, H, W)

def get_time_weights(T, device):
    weights = torch.arange(1, T + 1, device=device).float()
    weights = weights / weights.sum()  
    return weights.view(1, T, 1, 1, 1)

# ==========================================
# 2. 物理与保真度损失函数
# ==========================================
def pwv_weighted_mse_loss(pred, target, time_weights, spatial_mask, alpha=15.0):
    pwv = target[:, :, 0:1, :, :]
    pwv_norm = (pwv - pwv.min()) / (pwv.max() - pwv.min() + 1e-8)
    weight_map = 1.0 + alpha * pwv_norm
    
    sq_error = (pred[:, :, 1:2, :, :] - target[:, :, 1:2, :, :]) ** 2
    weighted_error = sq_error * weight_map * spatial_mask * time_weights
    
    return weighted_error.sum() / (weight_map * spatial_mask).sum()

def gaussian_window(window_size, sigma, channel):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

def ssim_loss(pred, target, spatial_mask, time_weights, window_size=11, sigma=1.5):
    pred_radar, target_radar = pred[:, :, 1:2], target[:, :, 1:2]
    B, T, C, H, W = pred_radar.shape
    
    pred_flat = pred_radar.reshape(B*T, C, H, W)
    target_flat = target_radar.reshape(B*T, C, H, W)
    window = gaussian_window(window_size, sigma, C).to(pred_flat.device)
    
    mu1 = F.conv2d(pred_flat, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(target_flat, window, padding=window_size//2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_flat*pred_flat, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target_flat*target_flat, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred_flat*target_flat, window, padding=window_size//2, groups=C) - mu1_mu2
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    ssim_map = ssim_map.view(B, T, C, H, W)
    weighted_ssim_loss = (1.0 - ssim_map) * spatial_mask * time_weights
    
    return weighted_ssim_loss.sum() / spatial_mask.sum()

def laplacian_divergence_loss(motion_fields):
    B, T, C, H, W = motion_fields.shape
    wind = motion_fields.reshape(B*T, C, H, W)
    u, v = wind[:, 0:1], wind[:, 1:2]
    
    u_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
    v_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
    u_dx = u_dx[:, :, 1:, :]  
    v_dy = v_dy[:, :, :, 1:]  
    div = u_dx + v_dy         
    
    lap_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=div.dtype, device=div.device)
    div_lap = F.conv2d(div, lap_kernel, padding=1)
    
    return torch.mean(torch.abs(div_lap))

def psi_harmonic_loss(psi):
    lap_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=psi.dtype, device=psi.device)
    B, T, C, H, W = psi.shape
    lap_psi = F.conv2d(psi.reshape(B*T, C, H, W), lap_kernel, padding=1, groups=C)
    return torch.mean(lap_psi ** 2)

# ==========================================
# 3. 训练主循环 (搭载 AMP 与 梯度累加)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = WeatherDataset('DATA_2025_S/PWV_2025_S', 'DATA_2025_S/RADAR_2025_S', seq_len_in=10, seq_len_out=20, img_size=(64, 64), sample_step=10)
    train_size = int(len(full_dataset) * 0.7)
    val_size = int(len(full_dataset) * 0.15)
    
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, train_size + val_size)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    model = WeatherPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 核心修复：更新为不带警告的新版 AMP GradScaler
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs('checkpoints', exist_ok=True)
    
    best_val_loss = float('inf')
    patience, no_improve = 10, 0
    num_epochs = 30
    accumulation_steps = 4 
    pred_steps = 20
    
    spatial_mask = create_spatial_mask(64, 64, device, sigma=15.0)
    time_weights = get_time_weights(pred_steps, device)
    
    print("开始 Phase 1: 非自回归物理基座训练 (搭载 AMP)...")
    
    try:
        for epoch in range(num_epochs):
            model.train() 
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs, motions, s_radar, s_max, psi_out, pwv_out = model(inputs, pred_steps=pred_steps)
                    
                    outputs = outputs.float()
                    motions = motions.float()
                    psi_out = psi_out.float()
                    
                    loss_pwv_mse = pwv_weighted_mse_loss(outputs, targets, time_weights, spatial_mask, alpha=15)
                    loss_ssim = ssim_loss(outputs, targets, spatial_mask, time_weights)
                    
                    loss_pwv_consist = nn.MSELoss()(pwv_out.float(), targets[:, :, 0:1])
                    loss_psi_harmonic = psi_harmonic_loss(psi_out)
                    loss_div_reg = laplacian_divergence_loss(motions)
                    
                    total_loss = (
                        0.60 * loss_pwv_mse +      
                        0.15 * loss_ssim +          
                        0.10 * loss_pwv_consist +   
                        0.05 * loss_psi_harmonic +  
                        0.10 * loss_div_reg         
                    ) / accumulation_steps
                
                scaler.scale(total_loss).backward()  
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                train_loss += total_loss.item() * accumulation_steps * inputs.size(0)
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            # ================= 验证环节 =================
            model.eval() 
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    with torch.autocast('cuda', dtype=torch.float16):
                        outputs, motions, _, _, psi_out, pwv_out = model(inputs, pred_steps=pred_steps)
                        outputs = outputs.float()
                        
                        batch_loss = (
                            0.60 * pwv_weighted_mse_loss(outputs, targets, time_weights, spatial_mask) +
                            0.15 * ssim_loss(outputs, targets, spatial_mask, time_weights) +
                            0.10 * nn.MSELoss()(pwv_out.float(), targets[:, :, 0:1]) +
                            0.05 * psi_harmonic_loss(psi_out.float()) +
                            0.10 * laplacian_divergence_loss(motions.float())
                        )
                    val_loss += batch_loss.item() * inputs.size(0)
                    
            avg_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"验证集 Loss 连续 {patience} 轮未改善，触发早停！")
                    break
                    
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'checkpoints/interrupted_model.pth')
        print("训练被手动中断，已保存当前模型权重。")

if __name__ == "__main__":
    main()