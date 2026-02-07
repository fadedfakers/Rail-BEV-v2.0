import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        # 更新门和重置门
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, 
                                    kernel_size, padding=padding)
        # 候选隐藏状态
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 
                                  kernel_size, padding=padding)

    def forward(self, input_tensor, h_cur):
        """
        input_tensor: 当前时刻特征 [B, C_in, H, W]
        h_cur: 上一时刻记忆 [B, C_hidden, H, W]
        """
        # 如果是序列第一帧，初始化 h_cur 为 0
        if h_cur is None:
            h_cur = torch.zeros(input_tensor.size(0), self.hidden_dim, 
                                input_tensor.size(2), input_tensor.size(3),
                                device=input_tensor.device, dtype=input_tensor.dtype)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        
        combined_new = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cnm = self.conv_can(combined_new)
        h_tilde = torch.tanh(cnm)
        
        h_next = (1 - update_gate) * h_cur + update_gate * h_tilde
        return h_next

class TemporalFusion(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, frames_num=4):
        super().__init__()
        self.frames_num = frames_num
        self.use_gru = True
        
        # SOTA 方案: ConvGRU
        self.gru = ConvGRUCell(input_dim=in_channels, hidden_dim=out_channels)
        
        # 降维/融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Input x: [B * T, C, H, W] (Batch 中混合了时序)
        Output: [B, C, H, W] (融合后的当前帧特征)
        """
        # 1. 恢复维度 [B, T, C, H, W]
        # 注意: 假设 Batch Size 是 B，输入是 B*T
        # 这里需要根据实际 dataloader 的逻辑调整，这里假设 input batch size 包含了 T
        bt, c, h, w = x.shape
        batch_size = bt // self.frames_num
        t = self.frames_num
        
        x = x.view(batch_size, t, c, h, w)
        
        # 2. 时序循环融合 (Past -> Current)
        hidden_state = None
        for i in range(t):
            # 获取第 i 帧 (0是主要过去帧, t-1是当前帧, 取决于 adapter 顺序)
            # 假设 adapter 顺序是 [t-3, t-2, t-1, t] (时间正序)
            feature_t = x[:, i, :, :, :] 
            hidden_state = self.gru(feature_t, hidden_state)
            
        # 3. 输出最终融合特征
        out = self.fusion_conv(hidden_state)
        return out