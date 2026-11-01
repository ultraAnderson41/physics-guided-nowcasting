import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset import WeatherDataset
from model import WeatherPredictor
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
import os

# ==========================================
# 物理反演：灰度 -> dBZ -> 降水率 (mm/h)
# ==========================================
def denormalize_to_rain_rate(tensor_data):
    tensor_data = np.clip(tensor_data, 0.0, 1.0)
    dbz = tensor_data * 50.0  
    z = 10.0 ** (dbz / 10.0)
    rain_rate = (z / 200.0) ** (1.0 / 1.6)
    rain_rate[tensor_data <= 0.01] = 0.0 
    return rain_rate

# ==========================================
# 图像质量指标
# ==========================================
def calc_ssim(true, pred, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1 = gaussian_filter(true, sigma=1.5)
    mu2 = gaussian_filter(pred, sigma=1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = gaussian_filter(true ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(pred ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(true * pred, sigma=1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

# ==========================================
# 气象评分指标 (严格像素级)
# ==========================================
def calc_pixel_metrics(true_mmh, pred_mmh, threshold_mmh):
    """严格的逐像素对比，无空间容错"""
    true_b = (true_mmh >= threshold_mmh)
    pred_b = (pred_mmh >= threshold_mmh)
    
    hits = np.sum(true_b & pred_b)
    misses = np.sum(true_b & ~pred_b)
    false_alarms = np.sum(~true_b & pred_b)
    
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    
    return csi, pod, far

# ==========================================
# 气象评分指标 (空间容错邻域级)
# ==========================================
def calc_neighborhood_metrics(true_mmh, pred_mmh, threshold_mmh, window_size=5):
    """使用最大值滤波器实现邻域空间容错评估 (Neighborhood CSI)"""
    true_b = (true_mmh >= threshold_mmh)
    pred_b = (pred_mmh >= threshold_mmh)
    
    true_n = maximum_filter(true_b, size=window_size)
    pred_n = maximum_filter(pred_b, size=window_size)
    
    hits = np.sum(true_n & pred_n)
    misses = np.sum(true_n & ~pred_n)
    false_alarms = np.sum(~true_n & pred_n)
    
    ncsi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    npod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    nfar = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    
    return ncsi, npod, nfar

def calc_fss(true_mmh, pred_mmh, threshold_mmh, window_size=5):
    """Fractions Skill Score (FSS)"""
    true_b = (true_mmh >= threshold_mmh).astype(np.float32)
    pred_b = (pred_mmh >= threshold_mmh).astype(np.float32)
    
    true_fraction = uniform_filter(true_b, size=window_size, mode='constant', cval=0.0)
    pred_fraction = uniform_filter(pred_b, size=window_size, mode='constant', cval=0.0)
    
    mse_fractions = np.mean((pred_fraction - true_fraction) ** 2)
    mse_ref = np.mean(pred_fraction ** 2 + true_fraction ** 2)
    
    if mse_ref == 0: return 0.0
    return 1.0 - (mse_fractions / mse_ref)

# ==========================================
# 空间功率谱密度 (PSD)
# ==========================================
def calc_psd(image):
    ftransform = np.fft.fft2(image)
    fshift = np.fft.fftshift(ftransform)
    magnitude_spectrum = np.abs(fshift)**2
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========== 启动全量国际标准基准测试 ==========")
    
    # 1. 加载数据集
    full_dataset = WeatherDataset(
        pwv_dir='DATA_2025_S/PWV_2025_S',
        radar_dir='DATA_2025_S/RADAR_2025_S',
        seq_len_in=10,
        seq_len_out=20,
        img_size=(64, 64),
        sample_step=5
    )
    
    total_samples = len(full_dataset)
    if total_samples == 0: 
        print("未找到有效数据！")
        return

    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    test_indices = list(range(train_size + val_size, total_samples))
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2. 加载模型
    model = WeatherPredictor(in_steps=10, out_steps=20).to(device)
    model_path = 'checkpoints_gan/best_gan_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("已成功加载预训练权重。")
    else:
        print("警告: 未找到预训练权重，将使用随机初始化权重进行测试。")
    model.eval()

    # 3. 指标阈值设定
    primary_th = 10.0 # 用于展示综合大盘的基准阈值 (大雨)
    multi_thresholds = [1.0, 5.0, 10.0, 20.0]
    
    metrics_keys = ['mse', 'copy_mse', 'ssim', 'copy_ssim', 
                    'csi', 'pod', 'far', 'copy_csi', 'copy_pod', 'copy_far',
                    'ncsi', 'npod', 'nfar', 'copy_ncsi', 'copy_npod', 'copy_nfar', 
                    'fss', 'copy_fss']
    
    frame_metrics = {i: {k: [] for k in metrics_keys} for i in range(20)}
    for th in multi_thresholds:
        for i in range(20):
            frame_metrics[i][f'multi_csi_{th}'] = []
            frame_metrics[i][f'multi_ncsi_{th}'] = []
            
    psd_true_list, psd_pred_list = [], []

    print(f"\n开始逐帧计算综合指标 (严格级 & 容错级)...")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 适配新的双场闭合模型的返回值解包
            # 统一修改成接收 6 个返回值，只取第一个 outputs
            outputs, _, _, _, _, _ = model(inputs, pred_steps=20)
            B = inputs.shape[0]
            
            for b in range(B):
                last_frame = inputs[b, -1, 1].cpu().numpy()  
                
                for t in range(20):
                    pred = outputs[b, t, 1].cpu().numpy()
                    true = targets[b, t, 1].cpu().numpy()
                    
                    frame_metrics[t]['mse'].append(np.mean((pred - true)**2))
                    frame_metrics[t]['copy_mse'].append(np.mean((last_frame - true)**2))
                    frame_metrics[t]['ssim'].append(calc_ssim(true, pred))
                    frame_metrics[t]['copy_ssim'].append(calc_ssim(true, last_frame))
                    
                    true_mmh = denormalize_to_rain_rate(true)
                    pred_mmh = denormalize_to_rain_rate(pred)
                    last_mmh = denormalize_to_rain_rate(last_frame)
                    
                    # 严格像素级指标
                    csi, pod, far = calc_pixel_metrics(true_mmh, pred_mmh, primary_th)
                    c_csi, c_pod, c_far = calc_pixel_metrics(true_mmh, last_mmh, primary_th)
                    
                    frame_metrics[t]['csi'].append(csi); frame_metrics[t]['copy_csi'].append(c_csi)
                    frame_metrics[t]['pod'].append(pod); frame_metrics[t]['copy_pod'].append(c_pod)
                    frame_metrics[t]['far'].append(far); frame_metrics[t]['copy_far'].append(c_far)

                    # 空间容错级指标
                    ncsi, npod, nfar = calc_neighborhood_metrics(true_mmh, pred_mmh, primary_th)
                    c_ncsi, c_npod, c_nfar = calc_neighborhood_metrics(true_mmh, last_mmh, primary_th)
                    fss = calc_fss(true_mmh, pred_mmh, primary_th)
                    c_fss = calc_fss(true_mmh, last_mmh, primary_th)
                    
                    frame_metrics[t]['ncsi'].append(ncsi); frame_metrics[t]['copy_ncsi'].append(c_ncsi)
                    frame_metrics[t]['npod'].append(npod); frame_metrics[t]['copy_npod'].append(c_npod)
                    frame_metrics[t]['nfar'].append(nfar); frame_metrics[t]['copy_nfar'].append(c_nfar)
                    frame_metrics[t]['fss'].append(fss); frame_metrics[t]['copy_fss'].append(c_fss)
                    
                    # 多阈值记录
                    for th in multi_thresholds:
                        th_csi, _, _ = calc_pixel_metrics(true_mmh, pred_mmh, th)
                        th_ncsi, _, _ = calc_neighborhood_metrics(true_mmh, pred_mmh, th)
                        frame_metrics[t][f'multi_csi_{th}'].append(th_csi)
                        frame_metrics[t][f'multi_ncsi_{th}'].append(th_ncsi)
                    
                    # 提取 T+60 分钟计算 PSD
                    if t == 9: 
                        psd_true_list.append(calc_psd(true))
                        psd_pred_list.append(calc_psd(pred))

    # ==========================================
    # 数据汇总与绘图 (图表 1 和 2)
    # ==========================================
    time_axis = [(i + 1) * 6 for i in range(20)]
    metrics = {k: [np.mean(frame_metrics[i][k]) for i in range(20)] for k in metrics_keys}

    # 图表 1: 严格指标 vs 容错指标对比面板
    fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10))
    plot_configs = [
        (0, 0, 'mse', 'copy_mse', 'MSE', 'blue'),
        (0, 1, 'ssim', 'copy_ssim', 'SSIM', 'green'),
        (0, 2, 'csi', 'copy_csi', f'Strict CSI @ {primary_th}mm/h', 'red'),
        (0, 3, 'pod', 'copy_pod', f'Strict POD @ {primary_th}mm/h', 'orange'),
        (1, 0, 'fss', 'copy_fss', f'FSS @ {primary_th}mm/h', 'purple'),
        (1, 1, 'nfar', 'copy_nfar', f'Neighb. FAR @ {primary_th}mm/h (Lower Better)', 'brown'),
        (1, 2, 'ncsi', 'copy_ncsi', f'Neighborhood CSI @ {primary_th}mm/h', 'darkred'),
        (1, 3, 'npod', 'copy_npod', f'Neighborhood POD @ {primary_th}mm/h', 'darkorange')
    ]
    
    for row, col, key_m, key_c, title, color in plot_configs:
        ax = axes1[row, col]
        ax.plot(time_axis, metrics[key_m], label='Model', marker='o', color=color)
        ax.plot(time_axis, metrics[key_c], label='Persistence (Copy)', linestyle='--', color='gray')
        ax.set_title(title)
        ax.set_xlabel('Lead Time (min)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    fig1.savefig('comprehensive_metrics.png', dpi=150)

    # 图表 2: 多阈值 CSI 演变与 PSD
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))
    colors = ['blue', 'green', 'orange', 'red']
    labels = [f'{th} mm/h' for th in multi_thresholds]
    
    for idx, th in enumerate(multi_thresholds):
        csi_trend = [np.mean(frame_metrics[i][f'multi_csi_{th}']) for i in range(20)]
        axes2[0].plot(time_axis, csi_trend, label=labels[idx], color=colors[idx], marker='o', markersize=4)
    axes2[0].set_title('Strict CSI Across Precipitation Levels')
    axes2[0].set_xlabel('Lead Time (min)'); axes2[0].grid(True, linestyle='--', alpha=0.6); axes2[0].legend()

    for idx, th in enumerate(multi_thresholds):
        ncsi_trend = [np.mean(frame_metrics[i][f'multi_ncsi_{th}']) for i in range(20)]
        axes2[1].plot(time_axis, ncsi_trend, label=labels[idx], color=colors[idx], marker='o', markersize=4)
    axes2[1].set_title('Neighborhood CSI Across Precipitation Levels')
    axes2[1].set_xlabel('Lead Time (min)'); axes2[1].grid(True, linestyle='--', alpha=0.6); axes2[1].legend()

    if len(psd_true_list) > 0:
        avg_psd_true = np.mean(np.array(psd_true_list), axis=0)
        avg_psd_pred = np.mean(np.array(psd_pred_list), axis=0)
        frequencies = np.arange(1, len(avg_psd_true))
        
        axes2[2].plot(frequencies, 10 * np.log10(avg_psd_true[1:]), label='Ground Truth', color='black', linestyle='--')
        axes2[2].plot(frequencies, 10 * np.log10(avg_psd_pred[1:]), label='Model Prediction', color='blue')
        axes2[2].set_title('Power Spectral Density (PSD) at T+60 min')
        axes2[2].set_xlabel('Spatial Frequency (1/wavelength)')
        axes2[2].set_ylabel('Power (dB)')
        axes2[2].grid(True, linestyle='--', alpha=0.6)
        axes2[2].legend()
        
    plt.tight_layout()
    fig2.savefig('csi_and_psd_analysis.png', dpi=150)

    # ==========================================
    # 动态挑选强降水样本进行图像直接对比 (功能完整保留)
    # ==========================================
    best_idx, max_storm_score = 0, 0
    search_range = min(300, len(test_dataset))
    for i in range(search_range):
        _, tgt = test_dataset[i]
        true_mmh = denormalize_to_rain_rate(tgt[:, 1].numpy())
        storm_score = np.sum(true_mmh >= 10.0) 
        if storm_score > max_storm_score:
            max_storm_score = storm_score
            best_idx = i

    sample_inputs, sample_targets = test_dataset[best_idx]
    with torch.no_grad():
        # 同步修改为接收 6 个返回值
        sample_outputs, _, _, _, _, _ = model(sample_inputs.unsqueeze(0).to(device), pred_steps=20)
    
    pred_imgs = sample_outputs.squeeze(0).cpu().numpy()
    true_imgs = sample_targets.numpy()
    last_input_radar = sample_inputs[-1, 1].numpy() 
    
    # 绘制 3: 3x5 序列直接对比图
    fig3, axes3 = plt.subplots(3, 5, figsize=(20, 12))
    fig3.suptitle(f'2-Hour Prediction Visualization (Sample #{best_idx})', fontsize=16)
    plot_indices = [3, 7, 11, 15, 19]
    labels = ['24 min', '48 min', '72 min', '96 min', '120 min']

    for col, idx in enumerate(plot_indices):
        axes3[0, col].imshow(pred_imgs[idx, 1], cmap='gray_r', vmin=0, vmax=1)
        axes3[0, col].set_title(f'Pred (+{labels[col]})'); axes3[0, col].axis('off')
        
        axes3[1, col].imshow(true_imgs[idx, 1], cmap='gray_r', vmin=0, vmax=1)
        axes3[1, col].set_title(f'True (+{labels[col]})'); axes3[1, col].axis('off')
        
        axes3[2, col].imshow(last_input_radar, cmap='gray_r', vmin=0, vmax=1)
        axes3[2, col].set_title('Last Input (Persistence)'); axes3[2, col].axis('off')

    plt.tight_layout()
    fig3.savefig('sequence_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 测试结束。全部评估图表及图像对比已保存到当前目录。")

if __name__ == "__main__":
    main()
