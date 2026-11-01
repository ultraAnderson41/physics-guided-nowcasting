import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataset import WeatherDataset
from torch.utils.data import DataLoader, Subset
from model import WeatherPredictor, SpatiotemporalDiscriminator

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

# ==========================================
# 3. 对抗训练辅助函数
# ==========================================
def prepare_discriminator_inputs(radar_seq, initial_radar_frame):
    """
    将序列展平，错位生成 prev_frame 用于计算时序差分
    输入 radar_seq: [B, T, 1, H, W]
    输出 current_frames: [B*T, 1, H, W], prev_frames: [B*T, 1, H, W]
    """
    B, T, C, H, W = radar_seq.shape
    curr = radar_seq.reshape(B*T, C, H, W)
    prev_seq = torch.cat([initial_radar_frame.unsqueeze(1), radar_seq[:, :-1, :, :, :]], dim=1)
    prev = prev_seq.reshape(B*T, C, H, W)
    return curr, prev

# ==========================================
# 4. Phase 2 训练主循环 (PatchGAN 博弈)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = WeatherDataset('DATA_2025_S/PWV_2025_S', 'DATA_2025_S/RADAR_2025_S', seq_len_in=10, seq_len_out=20, img_size=(64, 64), sample_step=10)
    train_size = int(len(full_dataset) * 0.7)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    net_G = WeatherPredictor().to(device)
    net_D = SpatiotemporalDiscriminator().to(device)
    
    # 强制加载 Phase 1 权重
    phase1_ckpt = 'checkpoints/best_model.pth'
    if os.path.exists(phase1_ckpt):
        net_G.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        print(f"成功加载 Phase 1 预训练物理基座: {phase1_ckpt}")
    else:
        raise FileNotFoundError("未找到 Phase 1 权重，对抗训练必须建立在物理基座之上！")

    # 优化器配置：冻结/压低物理推演头，放开底层特征
    unet_output_params = list(net_G.unet.outc.parameters())
    base_params = [p for n, p in net_G.named_parameters() if 'unet.outc' not in n]
    
    opt_G = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-4},
        {'params': unet_output_params, 'lr': 1e-6}
    ], weight_decay=1e-4)
    
    opt_D = torch.optim.AdamW(net_D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')
    
    os.makedirs('checkpoints_gan', exist_ok=True)
    
    num_epochs = 20
    pred_steps = 20
    spatial_mask = create_spatial_mask(64, 64, device, sigma=15.0)
    time_weights = get_time_weights(pred_steps, device)
    
    print("开始 Phase 2: PatchGAN 物理条件对抗网络训练...")
    
    try:
        for epoch in range(num_epochs):
            net_G.train()
            net_D.train()
            train_loss_G, train_loss_D = 0.0, 0.0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                initial_frame = inputs[:, -1, 1:2, :, :] 
                target_radar_seq = targets[:, :, 1:2, :, :] 
                
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs, motions, _, _, _, _ = net_G(inputs, pred_steps=pred_steps)
                    pred_radar_seq = outputs[:, :, 1:2, :, :]
                
                outputs_float = outputs.float()
                pred_radar_seq_float = pred_radar_seq.float()
                motions_float = motions.float()
                
                real_curr, real_prev = prepare_discriminator_inputs(target_radar_seq, initial_frame)
                fake_curr, fake_prev = prepare_discriminator_inputs(pred_radar_seq_float.detach(), initial_frame)

                # ---------------------
                # 训练 Discriminator
                # ---------------------
                opt_D.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    # d_out_real 和 d_out_fake 是 [B*T, 1, 7, 7] 的 Patch 矩阵
                    d_out_real = net_D(real_curr, real_prev)
                    d_out_fake = net_D(fake_curr, fake_prev)
                    
                    # 矩阵级 Hinge Loss: 要求真实 Patch 得分 > 1, 假 Patch 得分 < -1
                    loss_d_real = torch.mean(F.relu(1.0 - d_out_real))
                    loss_d_fake = torch.mean(F.relu(1.0 + d_out_fake))
                    loss_D = loss_d_real + loss_d_fake

                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()
                
                # ---------------------
                # 训练 Generator
                # ---------------------
                opt_G.zero_grad(set_to_none=True)
                fake_curr_for_G, fake_prev_for_G = prepare_discriminator_inputs(pred_radar_seq, initial_frame)
                
                with torch.autocast('cuda', dtype=torch.float16):
                    d_out_fake_for_G = net_D(fake_curr_for_G, fake_prev_for_G)
                    
                    # 对抗损失：使得生成的 Patch 在判别器中得分极高
                    loss_G_adv = -torch.mean(d_out_fake_for_G)
                    
                    loss_pwv_mse = pwv_weighted_mse_loss(outputs_float, targets, time_weights, spatial_mask, alpha=15)
                    loss_ssim = ssim_loss(outputs_float, targets, spatial_mask, time_weights)
                    loss_div_reg = laplacian_divergence_loss(motions_float)
                    
                    lambda_adv = 0.05
                    loss_G = (
                        0.60 * loss_pwv_mse + 
                        0.20 * loss_ssim + 
                        0.10 * loss_div_reg + 
                        lambda_adv * loss_G_adv
                    )
                
                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=5.0)
                
                scaler_G.step(opt_G)
                scaler_G.update()
                
                train_loss_G += loss_G.item() * inputs.size(0)
                train_loss_D += loss_D.item() * inputs.size(0)

            avg_loss_G = train_loss_G / len(train_loader.dataset)
            avg_loss_D = train_loss_D / len(train_loader.dataset)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss G: {avg_loss_G:.5f} | Loss D: {avg_loss_D:.5f}")
            
            torch.save(net_G.state_dict(), f'checkpoints_gan/gan_model_epoch_{epoch+1}.pth')
            torch.save(net_G.state_dict(), 'checkpoints_gan/best_gan_model.pth')
            
    except KeyboardInterrupt:
        torch.save(net_G.state_dict(), 'checkpoints_gan/interrupted_gan_model.pth')
        print("GAN 训练被中断，权重已保存。")

if __name__ == "__main__":
    main()