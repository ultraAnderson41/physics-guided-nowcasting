# 实际架构与数据流
唯一核心实现src/legacy/model.py。WeatherPredictor.forward先取PWV=input[:,:,0]、Radar=input[:,:,1]，随后将全部10帧Radar再全部10帧PWV拼接成[B,20,H,W]（不是时间交错通道）。UNet四次MaxPool、四次ConvTranspose，DoubleConv为3×3 Conv无bias→GroupNorm(8)→GELU，两遍。通道64→128→256→512→512；上采样拼接768/384/192/128，再压回256/128/64/64；outc 1×1输出80。
每连续4通道对应一个预报步[psi,chi,radar_raw,pwv_raw]。网络一次产生20步参数；PhysicsWarper从最后一帧两个场开始递推，后一帧依赖前一帧状态，网络本身不反复重编码预测帧。未实现ERA5环境或扩散条件输入。
返回tuple六组：[预测B,T,2,H,W；运动B,T,2,H,W；Radar有效生消B,T,1,H,W；cap B,T,1,H,W；psi B,T,1,H,W；PWV未来B,T,1,H,W]。T=20、H=W=64，预测通道仍PWV/Radar。
判别器只消费Radar当前帧和上一帧，内部拼[current,current-prev]，三次stride2卷积与谱归一化、InstanceNorm（首块无）、LeakyReLU后1通道7×7 patch评分。它不是生成器推理必需模块，训练中展平B*T。
