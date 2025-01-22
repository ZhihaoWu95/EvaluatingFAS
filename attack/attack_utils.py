from xml.dom import NoModificationAllowedErr
from lib import *
from color_model import color_correction
import yaml
sys.path.append('/mnt/woozh/SpoofingCFAS')

def load_img_batch(names:str, dir:str, device):
    pil2tensor = tv.transforms.ToTensor()
    imgs = [pil2tensor(Image.open(os.path.join(dir, name))).to(device) for name in names]
    return torch.stack([(img) for img in imgs])


def extract_multi_embeddings(nets, imgs, size = 112):
    pre_process1 = tv.transforms.Compose([
        tv.transforms.Resize((size,size)),
        tv.transforms.Normalize( mean=[0., 0., 0.], std=[1., 1., 1.]),
    ])

    img_align1 = torch.stack([pre_process1(img) for img in imgs])
    embeddings1 = torch.stack([net(img_align1) for net in nets[:2]])
    return F.normalize(embeddings1,p=2,dim=2)



def cross_multi_smooth_similarity_multi_targets(embeddings1,embeddings2, thre_en=False, threshold=None, weight=None):
    embeddings1 = embeddings1.unsqueeze(2)
    embeddings2 = embeddings2.unsqueeze(1)
    simis = torch.einsum('ijkl,ijkl->ijk', embeddings1, embeddings2).reshape(embeddings1.shape[0], -1)
    sim2prob = Sim2Prob(mode="leakyrelu", mid_thres=None, end_thres=-0.4, mid_prob=0.9, dynamic=weight)
    sim2prob_simis = sim2prob(simis, thre_en, threshold)

    if weight==None:
        sim2prob_simis_re = sim2prob_simis.mean()
    else:
        sim2prob_simis_re = sim2prob_simis.mean()

    return sim2prob_simis_re, simis


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


class Sim2Prob(nn.Module):
    def __init__(self,
                 mode: str = "relu",
                 mid_thres=None,
                 end_thres=-1,
                 mid_prob=0.9,
                 dynamic=False) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.mid = mid_thres
        self.end = end_thres
        self.mid_val = mid_prob
        self.dynamic_threashold = dynamic
        if self.mode == "relu":
            self.f = nn.ReLU()
        elif self.mode == "smoothl1":
            assert self.mid is not None
            self.f = nn.SmoothL1Loss(beta=mid_thres - end_thres,
                                     reduction="none")
        elif self.mode == "leakyrelu":
            self.f = nn.LeakyReLU(0.1)
        else:
            raise NotImplementedError

    def forward(self, sim, thre_en = False, threshold = None)->torch.Tensor:
        if self.mode == "relu":
            return -self.f(- sim - self.end)
        elif self.mode == "smoothl1":
            target = torch.full_like(sim, self.end)
            return 1 - self.f(sim, target) / (1 - 0.5 * (self.mid + self.end))
        elif self.mode == "leakyrelu":
            if thre_en == False:
                if self.dynamic_threashold is not None:
                    # min_index = torch.argmin(sim)
                    agg_simis = torch.mean(sim, dim=1)
                    min_index = torch.argmin(agg_simis)
                    self.dynamic_threashold[min_index][0] = 1.0
                    return sim * self.dynamic_threashold
                    # self.end = - (torch.min(sim) + 1e-3)
                return -self.f(- sim - self.end)
                # return 1/sim * - self.f(- sim - self.end)
            else:
                threshold_tensor = torch.tensor(threshold).to(sim.device)
                return -self.f(-sim + threshold_tensor.unsqueeze(1).repeat((1,sim.shape[1])))
        else:
            raise NotImplementedError


class Patch:
    def __init__(self,        
                 h: int,
                 w: int,
                 device: str = "cpu",
                 lr: float = 1 / 255,
                 momentum: float = 0.9,
                 opt_name: str = "MIFGSM",
                 is_mask = False,
                 mask_dir = None,
                 eot: bool = False,
                 eot_angle: float = math.pi / 15,
                 eot_scale: float = 0.85,
                 p: float = 0.8):
        self.w = int(w)
        self.h = int(h)
        self.eot = eot
        self.is_mask = is_mask
        self.shape = [1, 3, self.h, self.w]
        self.device = device
        self.opt_name = opt_name
        self.pil2tensor = tv.transforms.ToTensor()
        self.data = torch.rand(self.shape, device=device, requires_grad=True)
        self.g_momentum = torch.zeros_like(self.data, device=device).detach()
        self.data_previous = torch.zeros_like(self.data, device=device, requires_grad=True)
        self.grad_previous = torch.zeros_like(self.data, device=device).detach()
        if self.opt_name == "MIFGSM":
            self.opt = MIFGSM(m=momentum, lr=lr)
        elif self.opt_name == "NIFGSM":
            self.opt = NIFGSM(m=momentum, lr=lr)
        elif self.opt_name == "PIFGSM":
            self.opt = PIFGSM(m=momentum, lr=lr)

        # add
        self.data_ori = None

        # self.data = torch.ones(self.shape, device=device) * 0.5
        # self.data.requires_grad_()
        if is_mask:
            self.mask_patch = self.pil2tensor(Image.open(mask_dir))[0:3,:,:].to(self.device)
        if eot:
            
            self.robust = EoT(angle=eot_angle, scale=eot_scale, p=p)
        
    def apply(self,
              img: torch.Tensor,
              pos: Tuple[int, int],
              test_mode: bool = False,
              set_rotate: float = None,
              set_resize: float = None,) -> torch.Tensor:
        assert len(pos) == 2
        if self.eot:
            if test_mode:
                if self.is_mask:
                    switch, padding, _ = self.robust(self,
                                                    pos,
                                                    img.shape[-2:],
                                                    do_random_rotate=False,
                                                    do_random_color=False,
                                                    do_random_resize=False,
                                                    set_rotate=set_rotate,
                                                    set_resize=set_resize,
                                                    glass_mask=self.mask_patch)
                else:
                    switch, padding, _ = self.robust(self,
                                                    pos,
                                                    img.shape[-2:],
                                                    do_random_rotate=False,
                                                    do_random_color=False,
                                                    do_random_resize=False,
                                                    set_rotate=set_rotate,
                                                    set_resize=set_resize)
            else:
                if self.is_mask:
                    switch, padding, self.last_scale = self.robust(
                    self, pos, img.shape[-2:], do_random_rotate=True, glass_mask=self.mask_patch)
                # tv.utils.save_image(padding[0], '../1128/patch_test.png')
                else:
                    switch, padding, self.last_scale = self.robust(
                        self, pos, img.shape[-2:])
        else:
            switch, padding = self.mask(img.shape, pos)
        return (1 - switch) * img + switch * padding


    def savepatch(self,dir):
        if self.is_mask:  
            patch_content =  self.data
            patch_mask = torch.unsqueeze(self.mask_patch, dim=0)
            patch_content = patch_content * patch_mask
            patch_content = torch.where(patch_mask == 0, 1, patch_content)
            # hflip_image = torch.transforms.functional.hflip(patch_content)
            tv.utils.save_image(patch_content,dir)
        else:
            tv.utils.save_image(self.data,dir)

    def loadpatch(self,dir):
        data = self.pil2tensor(Image.open(dir).convert('RGB'))
        self.data = data.unsqueeze(0).to(self.device)
        self.data.requires_grad_()
        self.shape = list(self.data.shape)
        _, _, self.h, self.w = self.shape
        # add
        self.data_ori = self.pil2tensor(Image.open(dir).convert('RGB')).unsqueeze(0).to(self.device)


    def mask(self, shape: torch.Size,
             pos: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(shape, dtype=torch.float, device=self.device)
        if self.is_mask:
            mask[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = self.mask_patch
        else:
            mask[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = 1
        padding = torch.zeros(shape, dtype=torch.float, device=self.device)
        if self.is_mask:
            padding[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = self.data * self.mask_patch
        else:
            padding[..., pos[0]:pos[0] + self.h, pos[1]:pos[1] + self.w] = self.data
        return mask, padding

    def update(self, loss: torch.Tensor, dual = False) -> None:
        if dual == False:
            loss.backward()
        if self.opt_name == "MIFGSM":
            self.opt(self.data, self.data_ori)
        elif self.opt_name == "NIFGSM":
            self.g_momentum = self.opt(self.data, self.data_previous)
        elif self.opt_name == "PIFGSM":
            self.grad_previous = self.opt(self.data, self.data_previous, self.grad_previous)
        elif self.opt_name == "EMIFGSM":
            self.grad_previous = self.opt(self.data_n, self.data, self.data_previous)
        self.data.data.clamp_(0, 1)

    def random_pos(self, shape: torch.Size) -> Tuple[int, int]:
        h = random.randint(0, shape[-2] - self.h)
        w = random.randint(0, shape[-1] - self.w)
        return h, w


class EoT(nn.Module):
    def __init__(self, angle=math.pi / 18, scale=0.8, p=0.5, mask=None):
        super(EoT, self).__init__()
        self.angle = angle
        self.scale = scale
        self.p = p
        self.color = tv.transforms.ColorJitter(brightness=0.10)

    def forward(self,
                patch: Patch,
                pos: Tuple[int, int],
                img_shape: Tuple[int, int],
                do_random_rotate=True,
                do_random_color=True,
                do_random_resize=True,
                set_rotate=None,
                set_resize=None,
                glass_mask=None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        
        # do_random_rotate = False
        # do_random_color = False
        # do_random_resize = False
        
        if do_random_color:
            img_after_color = color_correction(patch.data,patch.data.device)
            # img = img_after_color
            img = self.color(img_after_color)
            # img = self.color(patch.data)
        else:
            img = patch.data

        if do_random_rotate:
            angle = torch.FloatTensor(1).uniform_(-self.angle, self.angle)
        elif set_rotate is None:
            angle = torch.zeros(1)
        else:
            angle = torch.full((1, ), set_rotate)

        
        pre_scale = 1 / (torch.cos(angle) + torch.sin(torch.abs(angle)))
        pre_scale = pre_scale.item()

        if do_random_resize:
            min_scale = min(self.scale / pre_scale, 1.0)
            scale_ratio = torch.FloatTensor(1).uniform_(min_scale, 1)
        elif set_resize is None:
            scale_ratio = torch.ones(1)
        else:
            scale_ratio = torch.full((1, ), set_resize)

        
        scale = scale_ratio * pre_scale
        logging.debug(
            f"scale_ratio: {scale_ratio.item():.2f}, "
            f"angle: {angle.item():.2f}, pre_scale: {pre_scale:.2f}, "
            f"scale: {scale.item():.2f}, ")

        t = -torch.ceil(torch.log2(scale))
        t = 1 << int(t.item())
        if t > 1:
            size = (patch.h // t, patch.w // t)
            img = F.interpolate(img, size=size, mode="area")
            scale *= t

        angle = angle.to(patch.device)
        scale = scale.to(patch.device)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.zeros((1, 2, 3), device=patch.device)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = 0
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = 0

        size = torch.Size((1, 3, patch.h // t, patch.w // t))
        grid = F.affine_grid(theta, size, align_corners=False)
        output = F.grid_sample(img, grid, align_corners=False)

        rotate_mask = torch.ones(size, device=patch.device)
        mask = F.grid_sample(rotate_mask, grid, align_corners=False)

        tw1 = (patch.w - patch.w // t) // 2
        tw2 = patch.w - patch.w // t - tw1
        th1 = (patch.h - patch.h // t) // 2
        th2 = patch.h - patch.h // t - th1

        pad = nn.ZeroPad2d(padding=(
            pos[1] + tw1,
            img_shape[1] - patch.w - pos[1] + tw2,
            pos[0] + th1,
            img_shape[0] - patch.h - pos[0] + th2,
        ))
        mask = pad(mask)
        padding = pad(output)
        mask = torch.clamp(mask, 0, 1)

        if glass_mask is not None:
            if len(glass_mask.shape) == 3:
                glass_mask = torch.unsqueeze(glass_mask, dim=0)
            glass_mask = F.grid_sample(glass_mask, grid, align_corners=False)
            glass_mask = pad(glass_mask)
            return glass_mask, padding, scale_ratio.item()
        
        return mask, padding, scale_ratio.item()


class MIFGSM(nn.Module):
    def __init__(self, m: float, lr: float):

        super().__init__()
        self.m = m
        self.lr = lr
        self.h = 0

    @torch.no_grad()
    def forward(self, t: torch.Tensor, ori_t: torch.Tensor) -> None:

        l1 = t.grad.abs().mean()
        if l1 == 0:
            l1 += 1
        self.h = self.h * self.m + t.grad / l1
        t.data -= self.lr * self.h.sign()
        # perb = t.data - self.lr * self.h.sign()
        if ori_t != None:
            update_data = torch.clamp(t.data - ori_t, -1, 1)
            t.data = update_data + ori_t
        else:
            update_data = 0
        
        # H constrain [166-235]
        t.data = torch.clamp(t.data, 0, 1)

        # print(perb - ori_t)
        # print(t.data)
        # print('- '*10)
        # t.data = update_data
        t.grad.zero_()

class NIFGSM(nn.Module):
    def __init__(self, m: float, lr: float):
        super().__init__()
        self.m = m
        self.lr = lr
        self.h = 0

    @torch.no_grad()
    def forward(self, t: torch.Tensor, pre_t: torch.Tensor):
        t_l1 = t.grad.abs().mean()
        if t_l1 == 0:
            t_l1 += 1
        self.h = self.h * self.m + t.grad / t_l1
        # pre_t.data = pre_t.data - self.lr * self.h.sign()
        # pre_t.data = torch.clamp(pre_t.data, 0, 1)
        t.data = pre_t.data - self.lr * self.h.sign()
        t.data = torch.clamp(t.data, 0, 1)
        t.grad.zero_()
        return self.h

class PIFGSM(nn.Module):
    def __init__(self, m: float, lr: float):
        super().__init__()
        self.m = m
        self.lr = lr
        self.h = 0

    @torch.no_grad()
    def forward(self, t: torch.Tensor, pre_t: torch.Tensor, pre_g: torch.Tensor):
        pre_g = t.grad
        t_l1 = pre_g.abs().mean()
        if t_l1 == 0:
            t_l1 += 1
        self.h = self.h * self.m + pre_g / t_l1
        t.data = pre_t.data - self.lr * self.h.sign()
        t.data = torch.clamp(t.data, 0, 1)
        t.grad.zero_()
        return pre_g
