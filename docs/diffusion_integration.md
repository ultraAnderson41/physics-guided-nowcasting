# 扩散模型协作协议（设计，尚未实现融合）
对方仓库 https://github.com/DYHLUCK/radar-pwv-diffusion/ 当前未能获取正文：网页抓取失败，公开GitHub API限流403。没有复制其实现，也不猜测它的网络/采样器。对接以本真实version1.0接口为准。

## 接口清单
| 张量/调用 | shape | 顺序/范围/单位/可微性 |
|---|---|---|
| physics_api.predict(model,x) 输入 | B,10,2,64,64 | float32、PWV/Radar、0..1；历史t−9..t0；不是[-1,1] |
| model.unet(cat(Radar历史,PWV历史)) | B,80,64,64 | 每4通道psi/chi/Radar raw/PWV raw，未限幅；可微；需自行构造相同通道拼接 |
| prediction | B,20,2,64,64 | PWV/Radar、0..1，t+6..120min（须确认实际采样）；可微，clamp外零梯度 |
| motion | B,20,2,64,64 | x东/右，y南/下，pixel/step，不是m/s；可微 |
| radar_source | B,20,1,64,64 | 有效正负生消，归一化Radar/step；替代门控梯度 |
| source_cap | B,20,1,64,64 | 正负共用cap，归一化经验量；声明可微但门控不给cap梯度 |
| psi | B,20,1,64,64 | 势函数代理，单位未物理标定，可微 |
| pwv_prediction | B,20,1,64,64 | 与prediction[:,:,0:1]相同，0..1可微 |
| warper.warp_frame(x,u,v) | B,C,64,64 | 可独立调用；需要指定场/位移；不是自动导出的全序列中间量 |

上述六组返回值已经直接存在，无须改核心；physics_api仅命名输出并检查形状。chi/rawPWV/rawRadar可以通过原unet单独算得，但不在Predictor六组返回中。原每步Radar_adv/PWV_adv和递推中间状态未直接输出；若深度耦合需要，可未来新增只读hook/可选diagnostics返回，必须逐元素回归；本轮没有增加hook或改返回格式。DEM在构造时加载，不额外输入UNet；没有ERA5环境输入。

## 候选方案比较
| 方案 | 最小变动/冻结 | 标签与传递 | 训练成本与主要风险 |
|---|---|---|---|
| A 后处理 | 另建扩散脚本，调用prediction，物理权重冻结 | 配对physics未来预测→真实未来Radar，可附PWV/历史；现有标签可用 | 单独训练扩散，多步采样额外成本；细节可能虚构强回波，应评FAR/强雨面积/位置 |
| B 条件生成 | 不改原代码，将历史+物理预测作条件，扩散输出未来Radar | condition仅来自起报历史及物理模型，不含目标；标签仍真实未来Radar | 可冻结物理先实现公平对照；需对方确认conditioning API和时空轴 |
| C 扩散残差 | 另建残差模块，基础预测冻结 | 标签target−physics_prediction可能负值且不在0..1；残差分布单独归一化 | 加回后clip可能扭曲极值/梯度，不能把残差当原图归一化；需保存scale与逆变换 |
| D 深层递推 | 改接口暴露adv/raw/状态，再评估联合训练 | 条件可含motion/source/adv；必须明确每步和轴，不把未来真值回灌 | 侵入和显存/梯度成本高；替代门控梯度不是真实物理导数，网络可能抵消物理约束 |

A/B/C可保持旧checkpoint不变，不必联合训练，也无需新增独立雨量标签（但评价仍为雷达经验雨强）。D若联合训练须单独新版本/新权重，不能声称历史原版相同。对90–120分钟是否保持强回波仅是研究假设，不是效果保证。随机生成要报告样本数量、采样seed、每成员和集合统计，禁止用真值挑最佳生成样本；同时报CSI/POD/FAR/FSS/MSE/面积偏差以及必要概率评分与可靠性。

## 推荐最小起点：冻结物理模型的B类条件输入
组员先调用physics_api.build_model(dem,checkpoint)，model.eval()，在torch.no_grad()内调用predict，取prediction[:,:,1:2]和历史Radar/PWV作条件。建议先只生成Radar，保留未来PWV作为可选物理条件，不误称其为额外观测。不要现在修改20→80网络或GAN。
导出协议应含sample_id、input/output timestamps、split、normalization、axis_order、物理checkpoint SHA、DEM SHA和源版本；condition与future_target分开命名。此导出器属于下一阶段，不在本轮偷加扩散实现。独立不带真值的合成forward可用于先验接口检查。
共同清单必须排除跨缺帧及分区共享帧（历史代码尚未做到），先登记这一必要新实验协议，Physics-only也在同样协议重评/重训后比较；不可拿旧Test图中的CSI与Hybrid直接比较。Diffusion-only与Hybrid使用相同历史可用信息、样本、训练/调参预算及相同生成成员数，保留物理only确定性作为基准。Test只在协议冻结后用一次；物理模型训练只用Train，不得从Val/Test权重选取对未来目标有泄漏的缓存。

## 请同组成员提供的最少信息
1. 可访问仓库或源码导出包、commit与许可证；当前参考链接未读到内容。
2. 真实forward/sample函数签名，B/T/C/H/W轴序、预测步数/历史步数、Radar或PWV通道要求。
3. 归一化[-1,1]还是[0,1]、预测epsilon/v/x0还是残差，采样调度与反归一化。
4. 条件注入方式、是否支持20帧64×64，成员数、推理seed/步数/显存和训练环境。
5. 已有Train/Val/Test清单、checkpoint/选模协议及指标定义，便于确认双方公平且无未来信息泄漏。
