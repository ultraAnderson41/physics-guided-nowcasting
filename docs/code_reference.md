# 文件索引与调用图

主调用：run_legacy.py→私有训练/评价模块.main→WeatherDataset / WeatherPredictor→UNet→PhysicsWarper→SpatialDerivatives / PhysicalGating；P2另外调用SpatiotemporalDiscriminator。

| 文件 | 类别/作用 | 关键符号（原行号） | 依赖 | 入口/纳入 |
|---|---|---|---|---|
| dataset.py | 同名配对、索引窗口、图像预处理 | WeatherDataset:7 | import os; import torch; from PIL import Image; from torch.utils.data import Dataset; from torchvision import transforms | 库接口，非main入口/字节一致纳入 |
| model.py | 核心模型/物理算子/UNet/PatchGAN | SpatialDerivatives:9, PhysicalGating:37, PhysicsWarper:55, DoubleConv:156, UNetGenerator:170, WeatherPredictor:217, SpatiotemporalDiscriminator:242 | import torch; import torch.nn as nn; import torch.nn.functional as F; import os | 库接口，非main入口/字节一致纳入 |
| test.py | 历史评价、PSD和绘图 | denormalize_to_rain_rate:14, calc_ssim:25, calc_pixel_metrics:41, calc_neighborhood_metrics:59, calc_fss:77, calc_psd:94, main:107 | import torch; import torch.nn as nn; import numpy as np; import matplotlib.pyplot as plt; from torch.utils.data import DataLoader, Subset; from dataset import WeatherDataset; from model import WeatherPredictor; from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter; import os | 原main入口（model/dataset为库）/字节一致纳入 |
| train_phase1.py | 损失函数和Phase1训练 | create_spatial_mask:13, get_time_weights:22, pwv_weighted_mse_loss:30, gaussian_window:40, ssim_loss:47, laplacian_divergence_loss:71, psi_harmonic_loss:87, main:96 | import torch; import torch.nn as nn; import torch.nn.functional as F; import math; import os; from dataset import WeatherDataset; from torch.utils.data import DataLoader, Subset; from model import WeatherPredictor | 原main入口（model/dataset为库）/字节一致纳入 |
| train_phase2.py | 重复损失定义和Hinge GAN训练 | create_spatial_mask:13, get_time_weights:22, pwv_weighted_mse_loss:30, gaussian_window:38, ssim_loss:45, laplacian_divergence_loss:67, prepare_discriminator_inputs:84, main:99 | import torch; import torch.nn as nn; import torch.nn.functional as F; import math; import os; from dataset import WeatherDataset; from torch.utils.data import DataLoader, Subset; from model import WeatherPredictor, SpatiotemporalDiscriminator | 原main入口（model/dataset为库）/字节一致纳入 |
| 图片结果/conv1.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/conv2.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/conv3.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/GAN_1.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/GAN_2.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/GAN_3.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase1_1.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase1_2.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase1_3.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase2_1_无地形.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase2_2无地形.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/phase2_3无地形.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/sample_step=5_1.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/sample_step=5_2.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 图片结果/sample_step=5_3.png | 历史结果图 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 地形数据/hebei_DEM.tfw | DEM源数据/地理元数据/归档 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 地形数据/hebei_DEM.tif | DEM源数据/地理元数据/归档 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 地形数据/hebei_DEM_30m分辨率_ASTER数据.tif.ovr | DEM源数据/地理元数据/归档 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 地形数据/hebei_DEM_30m分辨率_ASTER数据.tif.xml | DEM源数据/地理元数据/归档 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |
| 地形数据.zip | DEM源数据/地理元数据/归档 |  |  | 不纳入：external asset: size, provenance/privacy or sharing rights unconfirmed |

结构化完整清单见source_inventory.json：包含每个源文件签名、import、尺寸、SHA/未哈希策略、复制目标及排除理由。新增发布API/CLI是路径适配和保护，不是模型新机制。接口输入输出详见architecture与diffusion_integration。

## 类方法索引（原行号）

- src/legacy/dataset.py / WeatherDataset: __init__ (L8), __len__ (L59), __getitem__ (L62)
- src/legacy/model.py / SpatialDerivatives: __init__ (L11), dx (L23), dy (L30)
- src/legacy/model.py / PhysicalGating: forward (L40), backward (L47)
- src/legacy/model.py / PhysicsWarper: __init__ (L57), compute_s_max (L76), warp_frame (L94), forward (L107)
- src/legacy/model.py / DoubleConv: __init__ (L157), forward (L167)
- src/legacy/model.py / UNetGenerator: __init__ (L171), forward (L193)
- src/legacy/model.py / WeatherPredictor: __init__ (L218), forward (L223)
- src/legacy/model.py / SpatiotemporalDiscriminator: __init__ (L243), forward (L260)
