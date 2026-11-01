import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. 物理计算基础算子
# ==========================================
class SpatialDerivatives(nn.Module):
    """利用 Sobel 算子计算空间一阶导数（可导），用于提取风场和地形梯度"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]]) / 8.0
        
        self.register_buffer('weight_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('weight_y', sobel_y.view(1, 1, 3, 3))

    def dx(self, tensor):
        B, C, H, W = tensor.shape
        tensor = tensor.reshape(B * C, 1, H, W)
        pad = F.pad(tensor, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(pad, self.weight_x)
        return grad_x.reshape(B, C, H, W)

    def dy(self, tensor):
        B, C, H, W = tensor.shape
        tensor = tensor.reshape(B * C, 1, H, W)
        pad = F.pad(tensor, (1, 1, 1, 1), mode='replicate')
        grad_y = F.conv2d(pad, self.weight_y)
        return grad_y.reshape(B, C, H, W)

class PhysicalGating(torch.autograd.Function):
    """分段可导门控机制，前向硬截断，反向软传导"""
    @staticmethod
    def forward(ctx, s_raw, s_max):
        tau = 0.1 
        gate = torch.sigmoid((s_max - torch.abs(s_raw)) / tau)
        ctx.save_for_backward(gate)
        return torch.clamp(s_raw, -s_max, s_max)

    @staticmethod
    def backward(ctx, grad_output):
        gate, = ctx.saved_tensors
        grad_s_raw = grad_output * gate
        return grad_s_raw, None

# ==========================================
# 2. 物理推演引擎 (Physics Warper)
# ==========================================
class PhysicsWarper(nn.Module):
    """基于半拉格朗日平流和地形抬升方程的物理推演器"""
    def __init__(self, img_size=(64, 64)):
        super().__init__()
        self.img_size = img_size
        self.diff_ops = SpatialDerivatives()
        
        self.alpha = nn.Parameter(torch.tensor(1.0))   
        self.beta = nn.Parameter(torch.tensor(0.1))    
        self.oro_gamma = nn.Parameter(torch.tensor(0.5)) 
        
        dem_path = 'dem_64x64.pt'
        if os.path.exists(dem_path):
            dem_tensor = torch.load(dem_path)
            dem_norm = dem_tensor / (dem_tensor.max() + 1e-5)
            self.register_buffer('dem', dem_norm.view(1, 1, img_size[0], img_size[1]))
            print("PhysicsWarper: 成功挂载地形数据 (dem_64x64.pt)。")
        else:
            self.register_buffer('dem', torch.zeros(1, 1, img_size[0], img_size[1]))
            print("PhysicsWarper: 未找到 dem_64x64.pt，默认使用无地形平原模式。")

    def compute_s_max(self, u, v, pwv):
        B, C, H, W = pwv.shape
        
        flux_x = u * pwv
        flux_y = v * pwv
        div_flux = self.diff_ops.dx(flux_x) + self.diff_ops.dy(flux_y)
        
        dem_batch = self.dem.expand(B, -1, -1, -1)
        hx = self.diff_ops.dx(dem_batch)
        hy = self.diff_ops.dy(dem_batch)
        oro_lift = u * hx + v * hy
        
        moisture_conv = F.relu(-div_flux)
        orographic_forcing = F.relu(oro_lift) * pwv
        
        s_max = self.alpha * moisture_conv + self.beta * pwv + self.oro_gamma * orographic_forcing
        return s_max + 1e-4

    def warp_frame(self, x, u, v):
        B, C, H, W = x.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=x.device), 
                                        torch.arange(W, device=x.device), indexing='ij')
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1).float()
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1).float()

        x_norm = 2.0 * (x_grid - u.squeeze(1)) / (W - 1) - 1.0
        y_norm = 2.0 * (y_grid - v.squeeze(1)) / (H - 1) - 1.0

        grid = torch.stack((x_norm, y_norm), dim=-1)
        return F.grid_sample(x, grid, align_corners=True, padding_mode='border')

    def forward(self, radar_last, pwv_last, phys_params, future_steps=20):
        # 接收 [B, 1, H, W] 的初始观测帧
        radar_t = radar_last
        pwv_t = pwv_last
        
        preds, motions, s_radar_list, s_max_list, psi_list, pwv_out_list = [], [], [], [], [], []
        
        for t in range(future_steps):
            params_t = phys_params[:, t*4 : (t+1)*4]
            psi = params_t[:, 0:1]
            chi = params_t[:, 1:2]
            s_radar_raw = params_t[:, 2:3]
            s_pwv_raw = params_t[:, 3:4]
            
            psi_x, psi_y = self.diff_ops.dx(psi), self.diff_ops.dy(psi)
            chi_x, chi_y = self.diff_ops.dx(chi), self.diff_ops.dy(chi)
            u = -psi_y + chi_x
            v = psi_x + chi_y
            
            s_max = self.compute_s_max(u, v, pwv_t)
            
            s_radar_gated = PhysicalGating.apply(s_radar_raw, s_max)
            s_pwv_gated = PhysicalGating.apply(s_pwv_raw, s_max)
            
            radar_adv = self.warp_frame(radar_t, u, v)
            pwv_adv = self.warp_frame(pwv_t, u, v)
            
            radar_t = torch.clamp(radar_adv + s_radar_gated, 0.0, 1.0)
            pwv_t = torch.clamp(pwv_adv + s_pwv_gated, 0.0, 1.0)
            
            # 【核心修复】：按顺序拼接 PWV (通道0) 和 Radar (通道1)，打包成 [B, 2, H, W] 的标准格式
            preds.append(torch.cat([pwv_t, radar_t], dim=1))
            
            motions.append(torch.cat([u, v], dim=1)) 
            s_radar_list.append(s_radar_gated)
            s_max_list.append(s_max)
            psi_list.append(psi)
            pwv_out_list.append(pwv_t)
            
        return (torch.stack(preds, dim=1),
                torch.stack(motions, dim=1),
                torch.stack(s_radar_list, dim=1),
                torch.stack(s_max_list, dim=1),
                torch.stack(psi_list, dim=1),
                torch.stack(pwv_out_list, dim=1))

# ==========================================
# 3. 神经网络基座 (U-Net)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=20, out_channels=80): 
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512)) 

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(768, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(384, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(192, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))
        
        return self.outc(x)

# ==========================================
# 4. 生成器缝合体 (Predictor)
# ==========================================
class WeatherPredictor(nn.Module):
    def __init__(self, in_steps=10, out_steps=20):
        super().__init__()
        self.unet = UNetGenerator(in_channels=in_steps*2, out_channels=out_steps*4)
        self.warper = PhysicsWarper()

    def forward(self, inputs, pred_steps=20):
        # inputs 形状为 [B, T, 2, H, W]
        # 【核心修复】：保留通道维度，供 PhysicsWarper 中的初始帧提取
        pwv_seq = inputs[:, :, 0, :, :]   # [B, T, H, W]
        radar_seq = inputs[:, :, 1, :, :] # [B, T, H, W]
        
        # 拼给 U-Net 去算参数 [B, 20, H, W]
        x = torch.cat([radar_seq, pwv_seq], dim=1)
        phys_params = self.unet(x)
        
        # 提取最后一帧，并升维为 [B, 1, H, W] 喂给物理引擎做起点
        radar_last = radar_seq[:, -1:, :, :]
        pwv_last = pwv_seq[:, -1:, :, :]
        
        return self.warper(radar_last, pwv_last, phys_params, future_steps=pred_steps)

# ==========================================
# 5. 时空判别器 (PatchGAN for Phase 2)
# ==========================================
class SpatiotemporalDiscriminator(nn.Module):
    def __init__(self, in_channels=2): 
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1))]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, curr_frame, prev_frame):
        # 【核心修复】：接收当前帧和上一帧，在内部计算物理时序差分
        temp_diff = curr_frame - prev_frame
        
        # 将 当前帧 和 差分帧 在通道维度拼接，正好变成 in_channels=2 的输入张量
        x = torch.cat([curr_frame, temp_diff], dim=1)
        
        # 输出 [B*T, 1, 7, 7] 的 Patch 矩阵得分
        return self.model(x)