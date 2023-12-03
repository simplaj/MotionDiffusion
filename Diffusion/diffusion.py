import torch
import torch.nn as nn


class rotPosiEmb(nn.Module):
    """Some Information about rotPosiEmb"""
    def __init__(self, d: int, base: int = 10_000):
        super(rotPosiEmb, self).__init__()
        self.theta = nn.Parameter(1. / (base ** (torch.arange(0, d, 2).float() / d)), requires_grad=False)

    def forward(self, x):
        b, s, h, d = x.shape
        d_2 = d // 2
        
        seq_idx = torch.arange(s, device=x.device).type_as(self.theta)
        
        idx_theta = torch.einsum('n, d -> nd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        neg_half_x = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        
        rx = (x * idx_theta2.cos()[None, :, None, :]) + (neg_half_x * idx_theta2.sin()[None, :, None, :])

        return rx


class X_Attention(nn.Module):
    """Some Information about X_Attention"""
    def __init__(self, inp, hidden_dim, head_dim):
        super(X_Attention, self).__init__()
        self.inp = inp
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.pe = rotPosiEmb(hidden_dim // head_dim)
        self.softmax = nn.softmax(dim=-1)
        self.Q = nn.Linear(inp, hidden_dim)
        self.K = nn.Linear(inp, hidden_dim)
        self.V = nn.Linear(inp, hidden_dim)
        self.output = nn.Lineear(hidden_dim, inp)
        self.scale = (hidden_dim // head_dim) ** -0.5
        
    def forward(self, x, c):
        '''
        if use sa c=x else c=condition
        x:traj feature
        c:condition feature
        '''
        h_res = x
        att_shape = (*x.shape[:-1], self.head_dim, self.hidden_dim // self.head_dim)
        q, k, v = self.Q(c), self.K(x), self.V(x)
        q = q.view(att_shape)
        k = k.view(att_shape)
        v = v.view(att_shape)
        q = self.pe(q)
        k = self.pe(k)
        
        att_map = torch.einsum('bihd, bjhd -> bhij', q, v)
        att_map = self.softmax(att_map * self.scale)
        h = torch.einsum('bhij, bjhd -> bihd', att_map, v)
        h = h.reshape(*h.shape[:-2], -1)
        h = self.output(h)
        
        return h + h_res
    

class Attention_block(nn.Module):
    """Some Information about Attention_block"""
    def __init__(self, inp, hidden_dim, num_head):
        super(Attention_block, self).__init__()
        self.sa1 = X_Attention(inp, hidden_dim, num_head)
        self.sa2 = X_Attention(inp, hidden_dim, num_head)
        self.ca = X_Attention(inp, hidden_dim, num_head)
        self.norm1 = nn.LayerNorm(hidden_dim // num_head)
        self.norm2 = nn.LayerNorm(hidden_dim // num_head)
        self.norm3 = nn.LayerNorm(hidden_dim // num_head)

    def forward(self, x, c):
        x = self.sa1(x, x)
        x = self.norm1(x)
        x = nn.ReLU(x)
        x = self.sa2(x, x)
        x = self.norm2(x)
        x = nn.ReLU(x)
        x = self.ca(x, c)
        x = self.norm3(x)
        x = nn.ReLU(x)

        return x


class Denosier(nn.Module):
    """Some Information about Denosier"""
    def __init__(self, inp, oup, hidden_dim, num_head):
        super(Denosier, self).__init__()
        self.block1 = Attention_block(inp, hidden_dim, num_head)
        self.block2 = Attention_block(inp, hidden_dim, num_head)
        self.oup = nn.Linear(inp, oup)

    def forward(self, x, c):
        x = self.block1(x)
        x = self.block2(x)
        x = self.oup(x)

        return x


if __name__ == '__main__':
    rt = rotPosiEmb(64)
    a = torch.ones(32, 2, 8, 64)
    b = rt(a)
    print(b)