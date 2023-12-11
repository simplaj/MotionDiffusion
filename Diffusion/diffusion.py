import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import rff
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import reduce
from sklearn.decomposition import PCA


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
        self.softmax = nn.Softmax(dim=-1)
        self.Q = nn.Linear(inp, hidden_dim)
        self.K = nn.Linear(inp, hidden_dim)
        self.V = nn.Linear(inp, hidden_dim)
        self.output = nn.Linear(hidden_dim, inp)
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
        self.sa_blocks = nn.ModuleList()
        self.norm_blocks = nn.ModuleList()
        for i in range(4):
            self.sa_blocks.append(X_Attention(inp, hidden_dim, num_head))
            self.norm_blocks.append(nn.LayerNorm(inp))
        self.ca = X_Attention(inp, hidden_dim, num_head)
        self.norm_blocks.append(nn.LayerNorm(inp))

    def forward(self, x, c):
        for i in range(4):
            x = self.sa_blocks[i](x, c)
            x = self.norm_blocks[i](x)
            x = nn.ReLU()(x)
        x = self.ca(x, c)
        x = self.norm_blocks[-1](x)
        x = nn.ReLU()(x)

        return x


class Denosier(nn.Module):
    """Some Information about Denosier"""
    def __init__(self, inp, oup, hidden_dim, num_head):
        super(Denosier, self).__init__()
        self.proj1 = nn.Linear(138, inp)
        self.block1 = Attention_block(inp, hidden_dim, num_head)
        self.block2 = Attention_block(inp, hidden_dim, num_head)
        self.oup = nn.Linear(inp, oup)
        self.proj2 = nn.Linear(oup, 10)

    def forward(self, x, c, t):
        x = torch.cat([x, t], dim=-1)
        x = self.proj1(x)
        x = self.block1(x, c)
        x = self.block2(x, c)
        x = self.oup(x)
        x = self.proj2(x)

        return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m


class Diffusion(nn.Module):
    def __init__(
        self,
        pca=None,
        Net=Denosier,
        inp=256,
        oup=128,
        hidden_dim=1024,
        objective = 'pred_noise',
        num_head=8,
        timesteps=1000,
        ):
        super().__init__()
        self.model = Net(inp, oup, hidden_dim, num_head)
        self.objective = objective
        self.is_ddim_sampling = False
        self.pca_ = pca
        
        betas = linear_beta_schedule(timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        snr = alphas_cumprod / (1 - alphas_cumprod)
        
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
            
        register_buffer('loss_weight', loss_weight)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.rff = None
        self.num_joints = 17
        self.channels = 128
    
    @autocast(enable=False)
    def q_sample(self, x_start, t, noise):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def loss_fn(self, x0, con, t):
        b, n, c = x0.shape
        noise = torch.randn_like(x0)
        
        x = self.q_sample(x0, t, noise)
        self.rff = rff.layers.GaussianEncoding(sigma=10.0, input_size=b, encoded_size=64)
        t_embeding = self.rff(t)
        t_embeding = t_embeding.repeat(b, n, 1)
        
        out = self.model(x, con, t_embeding)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = F.mse_loss(out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def forward(self, x, con):
        b, n, c, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.loss_fn(x, con, t)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def pca(self, x):
        b, n, *_ = x.shape
        dtype = x.dtype
        x = x.reshape(n, -1)
        x = self.pca_.transform(x)
        x = torch.tensor(x, dtype=dtype).reshape(b, n, -1)
        return x
    
    def inverse_pca(self, x):
        b, n, *_= x.shape
        dtype = x.dtype
        x = self.pca_.inverse_transform(x)
        x = torch.tensor(x, dtype=dtype).reshape(b, n, *self.ori_shape[-2:])
        return x

    @torch.no_grad()
    def predict(self, x, c, mask, ori_shape):
        self.ori_shape = ori_shape
        y = torch.randn_like(x, device=x.device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            b, n, _ = y.shape
            t = torch.full((b,), t, device = x.device, dtype = torch.long)
            t_embeding = self.rff(t)
            t_embeding = t_embeding.repeat(b, n, 1)
            
            out = self.model(y, c, t_embeding)
            
            if self.objective == 'pred_noise':
                pred_noise = out
                y_de = self.predict_start_from_noise(y, t, pred_noise)
            
            y_no = self.q_sample(x, t, noise=None)
            
            y = self.pca(mask * self.inverse_pca(y_no) + (1 - mask) * self.inverse_pca(y_de))
            
        return y

    @torch.no_grad()
    def sample(self, batch_size=4):
        y = torch.randn((batch_size, self.num_joints, 10), device=self.betas.device)
        c = torch.ones((batch_size, self.num_joints, self.channels * 2), device=self.betas.device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            b, n, _ = y.shape
            t = torch.full((b,), t, device = self.betas.device, dtype = torch.long)
            self.rff = rff.layers.GaussianEncoding(sigma=10.0, input_size=b, encoded_size=64)
            t_embeding = self.rff(t)
            t_embeding = t_embeding.repeat(b, n, 1)
            
            out = self.model(y, c, t_embeding)
            
            if self.objective == 'pred_noise':
                pred_noise = out
                y = self.predict_start_from_noise(y, t, pred_noise)
            
        return y


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        diffusion: Diffusion,
        dataset: Dataset,
        ema_update_every = 10,
        ema_decay = 0.995,
        weight_decay=0.03,
        learning_rate=5e-4,
        train_batch_size=16,
        train_num_steps=100000,
        gradient_accumulate_every = 1,
        save_and_sample_every=1000,
        results_folder='./rst',
        amp=False,
        split_batches=True,
        mixed_precision_type='fp16',
        max_grad_norm = 1.
    ):
        super.__init__()
        
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        
        self.model = diffusion
        self.channels = self.model.channels
        
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, 
                        pin_memory=True, num_workers=cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        self.opt = AdamW(diffusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
            
        self.rst_folder = Path(results_folder)
        self.rst_folder.mkdir(exist_ok=True)
        
        self.step=0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
    @property
    def device(self):
        return self.accelerator.device
    
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
            
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            res = self.ema.ema_model.sample(batch_size=4)
                        torch.save(res, str(self.results_folder / f'sample-{milestone}.pt'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')     


def predict_example():
    x = torch.ones(1, 17, 10)
    c = torch.ones(1, 17, 256)
    pca = PCA(n_components=10)
    D = Diffusion()
    a = D(x, c)
    print(a)
    
    mask = torch.cat([torch.ones(5), torch.zeros(995)]).unsqueeze(-1)
    X = torch.ones(17, 1000, 3)
    ori_shape = X.shape
    X = X.reshape(17, -1)
    pca.fit(X)
    D.pca_ = pca
    x = pca.transform(X)
    x = torch.tensor(x, dtype=X.dtype).unsqueeze(0)
    a = D.predict(x, c, mask, ori_shape)
    print(a.shape)


def sample_example():
    d = Diffusion()
    a = d.sample()
    print(a)


if __name__ == '__main__':
    predict_example()
