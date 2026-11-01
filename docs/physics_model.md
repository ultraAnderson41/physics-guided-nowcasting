# 历史物理算子的实际公式
下列均来自src/legacy/model.py，不能用后期softsign门控替代解释。

| 类/方法 | 输入→输出 | 梯度/单位/限制 |
|---|---|---|
| SpatialDerivatives.dx/dy | BCHW→同形 | Sobel/8，replicate边界，可微；按像素差分，无米制网格距离 |
| PhysicsWarper.compute_s_max | u,v,PWV→cap | 参数被声明为Parameter，但门控反向不回传cap，原损失不直接惩罚cap，通常拿不到训练梯度 |
| PhysicalGating | raw,cap→clamped source | 自定义替代梯度，不是硬clamp真实导数，也不是softsign |
| warp_frame | 场和位移→平流场 | 双线性grid_sample、border、align_corners=True，对输入/位移分段可微 |

Sobel x为[[-1,0,1],[-2,0,2],[-1,0,1]]/8，y为其转置；图像+y向下。
u=−D_y psi+D_x chi；v=D_x psi+D_y chi。psi/chi为网络势函数代理，运动解释为每步像素位移，不是m/s。正u向右、正v向下。
DEM在构造时从CWD的dem_64x64.pt加载，h'=h/(max(h)+1e−5)，不提供CRS、缺测掩码或单位。没有文件时原版静默退化平地（发布API正式入口要求显式DEM）。
C=alpha*ReLU(−D_x(u*PWV)−D_y(v*PWV))+beta*PWV+gamma*ReLU(u*D_x h'+v*D_y h')*PWV+1e−4，初值alpha=1,beta=.1,gamma=.5。
cap并未clamp_min且参数本身无正约束；若人为改参数为负，可能出现非法上下界。这里是经验归一化许可，不是质量守恒方程或真实地形m/s抬升。PWV也是0..1代理，不可直接当kg/m²。
前向S=clamp(raw,−C,C)，Radar与PWV都用同一C（所以地形可影响正负生消，不仅增长）。反向对raw为grad_out*sigmoid((C−abs(raw))/.1)，对C返回None；对alpha/beta/gamma的间接梯度被切断，但它们改变C的数值仍会影响前向。饱和时raw梯度可以非常小。psi/chi仍通过平流获得梯度。
平流A(x)(i,j)=bilinear(x, j−u(i,j), i−v(i,j))；下一状态R'=clamp(A(R)+S_R,0,1)，P'=clamp(A(P)+S_P,0,1)。两步clamp均可能抹去修正。网络给出的20步raw和势各自不同，状态递推，未实现显式扩散、真实水汽质量闭合或边界通量约束。
