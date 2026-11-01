# 外部资产
本发布包不包含真实数据、DEM或权重。原dataset只读解压后的PWV/Radar PNG目录（月/日/同名PNG），不直接读ZIP。
将数据放在本目录或在configs/paths.example.json副本中填写个人绝对路径（不要将私有路径提交Git）。
dem_64x64.pt应为可信64×64 Tensor；其原始来源在version1.0目录之外，数值单位/配准未获该版本证明。不能把后期V2/ERA5资产直接解释为本历史模型原资产。
checkpoint应为与WeatherPredictor兼容的可信原始state_dict。缺少时正式推理/评价入口拒绝运行，合成示例不需要它。
原始地形ZIP/GeoTIFF/OVR、历史图像、ERA5均无需放入此发布仓库。本版本不依赖ERA5。
