import os
import math
import random
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs',
                     'probe_outputs', 'seg_outputs', 'grader_outputs', 'grounded_morph_outputs',
                     'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'clinical_multitask_outputs', 'uncertainty_outputs', 'vlm_outputs',
                     'explainability_outputs', 'figures', 'embryo_project'}   # NEW: consolidated project folder
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]
TOKEN_NAMES = ['ICM', 'TE', 'Blastocoel', 'Zona', 'Fragmentation', 'Global']

INPUT_DIR = "."
IMAGE_DIR = "./Downloads/archive/Images/Images"

CONFIG = {
    "dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3, "val_viz": 6, "seed": 42},
    "augmentation": {
        # ENHANCEMENT: same rationale as Phase 1 (embryo_mae_pretrain_v2.py) —
        # local zoomed-in crops teach the tokens fine sub-structure texture,
        # not just whole-embryo shape, which is what ICM/TE grading needs.
        "local_view_prob": 0.5,
        "local_crop_min_frac": 0.35,
        "local_crop_max_frac": 0.65,
    },
    "model": {"embed_dim": 768, "depth": 12, "num_heads": 12, "num_morphology_tokens": 6,
              "decoder_embed_dim": 512, "decoder_depth": 4, "decoder_num_heads": 16,
              "mask_ratio": 0.75, "norm_pix_loss": True, "lambda_div": 0.1},
    "training": {"batch_size": 16, "accum_iter": 1, "epochs": 250, "lr": 1.5e-4, "min_lr": 1e-6,
                 "weight_decay": 0.05, "warmup_ratio": 0.05, "grad_clip": 1.0,
                 "num_workers": 2, "pin_memory": True,
                 # consolidated: embryo_project/morphology/{figures,checkpoints}/
                 "output_dir": "./embryo_project/morphology"},
}

# ============================================================
# IMAGE DISCOVERY + DATASET  (global view + local zoomed-crop view)
# ============================================================
def find_images(root):
    for start in [root, os.getcwd(), os.path.expanduser("~")]:
        imgs = []
        if start and os.path.isdir(start):
            for dp, dirs, files in os.walk(start):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs')]
                for f in files:
                    if f.lower().endswith(IMAGE_EXTS):
                        imgs.append(os.path.join(dp, f))
        if imgs:
            return sorted(set(imgs))
    return []

def post_crop_transform(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class ImgDataset(Dataset):
    def __init__(self, paths, size, train, aug_cfg=None):
        self.paths = list(paths); self.size = size; self.train = train
        self.aug_cfg = aug_cfg or {}
        self.tf = post_crop_transform(size) if train else eval_tf(size)
    def __len__(self):
        return len(self.paths)
    def _local_zoom_crop(self, image):
        h, w = image.shape[:2]
        frac = random.uniform(self.aug_cfg.get('local_crop_min_frac', 0.35),
                              self.aug_cfg.get('local_crop_max_frac', 0.65))
        ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
        y0 = random.randint(0, max(0, h - ch)); x0 = random.randint(0, max(0, w - cw))
        return image[y0:y0 + ch, x0:x0 + cw]
    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        if self.train and random.random() < self.aug_cfg.get('local_view_prob', 0.5):
            img = self._local_zoom_crop(img)
        return self.tf(image=img)['image']

def get_fixed_viz_batch(image_paths, cfg, n=6):
    ds = ImgDataset(image_paths[:n], cfg['dataset']['image_size'], train=False)
    if len(ds) == 0:
        return torch.zeros(1, 3, cfg['dataset']['image_size'], cfg['dataset']['image_size'])
    return torch.stack([ds[i] for i in range(len(ds))])

# ============================================================
# POS EMBED + PATCH EMBED
# ============================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, num_tokens):
    gh = np.arange(grid_size, dtype=np.float32); gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape([2, 1, grid_size, grid_size])
    def _1d(d, pos):
        omega = np.arange(d // 2, dtype=np.float32); omega /= d / 2.0; omega = 1.0 / 10000 ** omega
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([_1d(embed_dim // 2, grid[0]), _1d(embed_dim // 2, grid[1])], axis=1)
    emb = np.concatenate([np.zeros([num_tokens, embed_dim], dtype=np.float32), emb], axis=0)
    return torch.from_numpy(emb).float()

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

# ============================================================
# GROUNDED MORPHOLOGY-MAE — architecture unchanged (already a proper
# ViT-Base: 768-dim, 12 real layers), only augmentation/epochs enhanced.
# ============================================================
class MorphologyMAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nt = cfg['model']['num_morphology_tokens']
        self.norm_pix_loss = cfg['model']['norm_pix_loss']
        ed = cfg['model']['embed_dim']; dd = cfg['model']['decoder_embed_dim']
        ps = cfg['dataset']['patch_size']; ic = cfg['dataset']['in_chans']
        self.patch_size = ps
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], ps, ic, ed)
        P = self.patch_embed.num_patches
        self.morph_tokens = nn.Parameter(torch.zeros(1, self.nt, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.nt + P, ed), requires_grad=False)
        el = nn.TransformerEncoderLayer(d_model=ed, nhead=cfg['model']['num_heads'], dim_feedforward=ed * 4,
                                        dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=cfg['model']['depth'], enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(ed)
        self.decoder_embed = nn.Linear(ed, dd, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.nt + P, dd), requires_grad=False)
        dl = nn.TransformerEncoderLayer(d_model=dd, nhead=cfg['model']['decoder_num_heads'], dim_feedforward=dd * 4,
                                        dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dl, num_layers=cfg['model']['decoder_depth'], enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(dd)
        self.decoder_pred = nn.Linear(dd, ps * ps * ic, bias=True)

        g = int(P ** 0.5)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(ed, g, self.nt).unsqueeze(0))
        self.decoder_pos_embed.data.copy_(get_2d_sincos_pos_embed(dd, g, self.nt).unsqueeze(0))
        nn.init.normal_(self.morph_tokens, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_size; h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p * p * 3)

    def random_masking(self, x, ratio):
        N, L, D = x.shape
        keep = int(L * (1 - ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_sh = torch.argsort(noise, dim=1)
        ids_res = torch.argsort(ids_sh, dim=1)
        ids_keep = ids_sh[:, :keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device); mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_res)
        return x_keep, mask, ids_res

    def forward_encoder(self, imgs, ratio):
        x = self.patch_embed(imgs) + self.pos_embed[:, self.nt:, :]
        x, mask, ids_res = self.random_masking(x, ratio)
        morph = (self.morph_tokens + self.pos_embed[:, :self.nt, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([morph, x], dim=1)
        x = self.enc_norm(self.encoder(x))
        return x[:, :self.nt, :], x[:, self.nt:, :], mask, ids_res

    def forward_decoder(self, morph_enc, patch_enc, ids_res):
        m = self.decoder_embed(morph_enc)
        p = self.decoder_embed(patch_enc)
        B, _, dd = p.shape
        n_mask = ids_res.shape[1] - p.shape[1]
        mtok = self.mask_token.repeat(B, n_mask, 1)
        p_full = torch.cat([p, mtok], dim=1)
        p_full = torch.gather(p_full, 1, ids_res.unsqueeze(-1).repeat(1, 1, dd))
        p_full = p_full + self.decoder_pos_embed[:, self.nt:, :]
        m = m + self.decoder_pos_embed[:, :self.nt, :]
        x = self.dec_norm(self.decoder(torch.cat([m, p_full], dim=1)))
        return self.decoder_pred(x[:, self.nt:, :])

    def recon_loss(self, imgs, pred, mask):
        tgt = self.patchify(imgs)
        if self.norm_pix_loss:
            mu = tgt.mean(-1, keepdim=True); var = tgt.var(-1, keepdim=True)
            tgt = (tgt - mu) / (var + 1e-6) ** 0.5
        loss = ((pred - tgt) ** 2).mean(-1)
        return (loss * mask).sum() / mask.sum()

    def diversity_loss(self, morph_enc):
        n = F.normalize(morph_enc, dim=-1)
        sim = torch.bmm(n, n.transpose(1, 2))
        eye = torch.eye(self.nt, device=morph_enc.device).unsqueeze(0)
        return ((sim - eye) ** 2).mean()

    def forward(self, imgs, ratio=0.75):
        morph_enc, patch_enc, mask, ids_res = self.forward_encoder(imgs, ratio)
        pred = self.forward_decoder(morph_enc, patch_enc, ids_res)
        recon = self.recon_loss(imgs, pred, mask)
        div = self.diversity_loss(morph_enc)
        total = recon + self.cfg['model']['lambda_div'] * div
        return total, recon, div, morph_enc

    @torch.no_grad()
    def encode(self, imgs):
        x = self.patch_embed(imgs) + self.pos_embed[:, self.nt:, :]
        morph = (self.morph_tokens + self.pos_embed[:, :self.nt, :]).expand(x.shape[0], -1, -1)
        x = self.enc_norm(self.encoder(torch.cat([morph, x], dim=1)))
        return x[:, :self.nt, :], x[:, self.nt:, :]

# ============================================================
# TRAIN + VIZ
# ============================================================
def lr_at(ef, cfg, warmup):
    b, mn, tot = cfg['training']['lr'], cfg['training']['min_lr'], cfg['training']['epochs']
    if ef < warmup:
        s = ef / max(1, warmup)
    else:
        s = 0.5 * (1 + math.cos(math.pi * (ef - warmup) / max(1, tot - warmup)))
    return mn + (b - mn) * s

@torch.no_grad()
def viz_token_attention(model, imgs, cfg, path, n=3, title=None):
    model.eval()
    dev = next(model.parameters()).device
    n = min(n, imgs.shape[0]); x = imgs[:n].to(dev)
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        morph, patch = model.encode(x)
    morph = F.normalize(morph.float(), dim=-1); patch = F.normalize(patch.float(), dim=-1)
    aff = torch.bmm(morph, patch.transpose(1, 2))
    g = cfg['dataset']['image_size'] // cfg['dataset']['patch_size']; s = cfg['dataset']['image_size']
    T = cfg['model']['num_morphology_tokens']
    mean = np.array(IMNET_MEAN).reshape(1, 3, 1, 1); std = np.array(IMNET_STD).reshape(1, 3, 1, 1)
    orig = np.clip(x.float().cpu().numpy() * std + mean, 0, 1).transpose(0, 2, 3, 1)
    fig, axes = plt.subplots(n, T + 1, figsize=(2.3 * (T + 1), 2.5 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(orig[i]); axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Input', fontsize=10)
        for k in range(T):
            m = aff[i, k].reshape(g, g).cpu().numpy()
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            m = cv2.resize(m.astype(np.float32), (s, s))
            axes[i, k + 1].imshow(orig[i]); axes[i, k + 1].imshow(m, cmap='jet', alpha=0.5); axes[i, k + 1].axis('off')
            if i == 0:
                axes[i, k + 1].set_title('Token {}'.format(k), fontsize=10)
    if title:
        fig.suptitle(title, fontsize=13); plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()

def train():
    cfg = CONFIG
    random.seed(cfg['dataset']['seed']); np.random.seed(cfg['dataset']['seed']); torch.manual_seed(cfg['dataset']['seed'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = cfg['training']['output_dir']
    fig_dir = os.path.join(out, 'figures'); ck_dir = os.path.join(out, 'checkpoints')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(ck_dir, exist_ok=True)

    root = IMAGE_DIR if (IMAGE_DIR and os.path.isdir(IMAGE_DIR)) else INPUT_DIR
    paths = find_images(root)
    if not paths:
        raise FileNotFoundError("No images found. Set IMAGE_DIR to the embryo .png folder.")
    from collections import Counter
    loc = Counter(os.path.dirname(p) for p in paths[:1000]).most_common(1)[0][0]
    print("[IMAGES] {} images | most under: {}\n".format(len(paths), loc))
    print("[WHOLE DATASET] using all {} discovered images (no subsampling)\n".format(len(paths)))
    print("[WHOLE DATASET] using all {} discovered images (no subsampling)\n".format(len(paths)))

    size = cfg['dataset']['image_size']
    rng = random.Random(cfg['dataset']['seed']); rng.shuffle(paths)
    n_viz = min(cfg['dataset']['val_viz'], len(paths))
    viz_paths, train_paths = paths[:n_viz], paths[n_viz:]
    viz_imgs = get_fixed_viz_batch(viz_paths, cfg, n=n_viz)

    loader = DataLoader(ImgDataset(train_paths, size, train=True, aug_cfg=cfg['augmentation']),
                        batch_size=cfg['training']['batch_size'], shuffle=True,
                        num_workers=cfg['training']['num_workers'],
                        pin_memory=cfg['training']['pin_memory'], drop_last=True)

    model = MorphologyMAE(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], betas=(0.9, 0.95),
                            weight_decay=cfg['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    epochs = cfg['training']['epochs']; accum = cfg['training']['accum_iter']
    warmup = max(1, int(cfg['training']['warmup_ratio'] * epochs)); steps = max(1, len(loader))
    viz_every = max(1, epochs // 6); ratio = cfg['model']['mask_ratio']

    print("[TRAIN] grounded morphology-MAE (v2, +multi-scale aug) | device={} | eff-batch={} | steps/epoch={} | epochs={}\n".format(
        dev, cfg['training']['batch_size'] * accum, steps, epochs))
    viz_token_attention(model, viz_imgs, cfg, os.path.join(fig_dir, 'attention_epoch_000_init.png'),
                        title='Before training (random init)')

    # RECORD: save the real boolean masking pattern used during pretraining
    # (which patches get hidden per forward pass) -- actual data, not just a picture.
    with torch.no_grad():
        _, _, mask_sample, _ = model.forward_encoder(viz_imgs[:min(6, viz_imgs.shape[0])].to(dev), ratio)
    np.save(os.path.join(fig_dir, 'masking_pattern_sample.npy'), mask_sample.cpu().numpy())
    print("[RECORD] saved real masking pattern array -> masking_pattern_sample.npy (shape {})\n".format(
        tuple(mask_sample.shape)))

    # RECORD: save the real boolean masking pattern used during pretraining
    # (which patches get hidden per forward pass) -- actual data, not just a picture.
    with torch.no_grad():
        _, _, mask_sample, _ = model.forward_encoder(viz_imgs[:min(6, viz_imgs.shape[0])].to(dev), ratio)
    np.save(os.path.join(fig_dir, 'masking_pattern_sample.npy'), mask_sample.cpu().numpy())
    print("[RECORD] saved real masking pattern array -> masking_pattern_sample.npy (shape {})\n".format(
        tuple(mask_sample.shape)))

    hist = {'total': [], 'recon': [], 'div': []}; best = float('inf')
    for ep in range(epochs):
        model.train(); rt = rr = rd = 0.0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(loader), total=steps, desc="Epoch {}/{}".format(ep + 1, epochs))
        for st, imgs in pbar:
            for pg in opt.param_groups:
                pg['lr'] = lr_at(ep + st / steps, cfg, warmup)
            imgs = imgs.to(dev, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                total, recon, div, _ = model(imgs, ratio)
                ls = total / accum
            scaler.scale(ls).backward()
            if (st + 1) % accum == 0 or (st + 1) == steps:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            rt += total.item(); rr += recon.item(); rd += div.item()
            pbar.set_postfix(total="{:.4f}".format(total.item()), recon="{:.4f}".format(recon.item()),
                             div="{:.4f}".format(div.item()))
        at, ar, ad = rt / steps, rr / steps, rd / steps
        hist['total'].append(at); hist['recon'].append(ar); hist['div'].append(ad)
        print("[Epoch {:03d}] total={:.4f} | recon={:.4f} | div={:.4f}".format(ep + 1, at, ar, ad))

        torch.save({'model': model.state_dict(), 'epoch': ep, 'loss': at, 'config': cfg},
                   os.path.join(ck_dir, 'morph_last.pth'))
        if at < best:
            best = at
            torch.save({'model': model.state_dict(), 'epoch': ep, 'loss': at, 'config': cfg},
                       os.path.join(ck_dir, 'morph_best.pth'))
        if (ep + 1) % viz_every == 0 or (ep + 1) == epochs:
            viz_token_attention(model, viz_imgs, cfg, os.path.join(fig_dir, 'attention_epoch_{:03d}.png'.format(ep + 1)),
                                title='Epoch {} (recon {:.4f})'.format(ep + 1, ar))

    plt.figure(figsize=(9, 5))
    e = range(1, epochs + 1)
    plt.plot(e, hist['total'], marker='o', label='Total'); plt.plot(e, hist['recon'], marker='s', label='Reconstruction')
    plt.plot(e, hist['div'], marker='^', label='Diversity')
    plt.title('Grounded morphology-MAE loss (v2, +multi-scale aug)'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'training_loss_curve.png'), dpi=200, bbox_inches='tight'); plt.close()

    # RECORD: real per-epoch loss values as data (CSV), not just the rendered curve
    import pandas as pd
    pd.DataFrame({'epoch': list(e), 'total_loss': hist['total'], 'recon_loss': hist['recon'],
                 'diversity_loss': hist['div']}).to_csv(os.path.join(out, 'training_history.csv'), index=False)

    # RECORD: real per-epoch loss values as data (CSV), not just the rendered curve
    import pandas as pd
    pd.DataFrame({'epoch': list(e), 'total_loss': hist['total'], 'recon_loss': hist['recon'],
                 'diversity_loss': hist['div']}).to_csv(os.path.join(out, 'training_history.csv'), index=False)
    viz_token_attention(model, viz_imgs, cfg, os.path.join(fig_dir, 'attention_final.png'), title='Final trained attention')
    print("\nDone. Checkpoint: {}".format(os.path.join(ck_dir, 'morph_best.pth')))

    try:
        from IPython.display import Image, display
        for fp in sorted(glob.glob(os.path.join(fig_dir, '*.png'))):
            print("Displaying:", os.path.basename(fp)); display(Image(filename=fp, width=820))
    except Exception:
        pass

if __name__ == '__main__':
    train()
