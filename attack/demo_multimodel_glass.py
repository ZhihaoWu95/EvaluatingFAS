import argparse

from torch import randint
from attack_utils import *
sys.path.append('/mnt/woozh/SpoofingCFAS')
from multi_model.model import get_models, get_threshold
import time
import torchvision.transforms.functional as transf

DEVICE = torch.device(0)
ALL_MODEL = np.array(range(0,3))
target_model = np.array(range(0,2))
OUT_MODEL = np.array([2])


def tv_loss(img):
    h, w = img.shape[-2], img.shape[-1]
    img_a = img[..., : h - 1, : w - 1]
    img_b = img[..., 1:, : w - 1]
    img_c = img[..., : h - 1, 1:]
    tv = ((img_a - img_b) ** 2 + (img_a - img_c) ** 2 + 1e-9) ** 0.5
    return tv.mean()

def ct_loss(img_patch_region, patch):
    h, w = img_patch_region.shape[-2], img_patch_region.shape[-1]
    img_a = img_patch_region[..., : h - 1, : w - 1]
    img_b = img_patch_region[..., 1:, : w - 1]
    img_c = img_patch_region[..., : h - 1, 1:]
    ct = ((patch - img_a) ** 2 + (patch - img_b) ** 2 + (patch - img_c) ** 2 + 1e-9) ** 0.5
    return ct.mean()

def min_max_norm(input_tensor):
    min_value = torch.min(input_tensor)
    max_value = torch.max(input_tensor)
    normalized_tensor = (input_tensor - min_value) / (max_value - min_value)
    return normalized_tensor

def ld_loss(ir_img, patch):
    h_patch, _ = torch.max(patch.data, dim=1)
    h_patch =  F.softmax(torch.flatten(h_patch))
    h_ir, _ = torch.max(ir_img, dim=1)
    h_ir =  F.softmax(torch.flatten(h_ir))
    ld_dist = F.kl_div(h_ir.log(), h_patch, reduction='sum')

    return ld_dist

def input_transform(x, iter):
    
    p3 = torch.rand(1).to(DEVICE)
    if p3 < 0.5:
        x = transf.adjust_sharpness(x, p3 * 6.4 + 0.8)

    p1 = torch.rand(1).to(DEVICE)
    if p1 < 0.5:
        x = torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)
    if p1 < 0.75 and p1 > 0.25:
        p1 = p1 - 0.25
        x = transf.adjust_gamma(x, 3*p1+0.5)
    # noise_size = np.random.randint(20)
    # x = torch.clip(x + torch.zeros_like(x).uniform_(-noise_size/255,noise_size/255), 0, 1)
    return x

def multiple_cross_simi_cal(embeddings1, embeddings2, dynamic_threashold):
    simis = (embeddings1 * embeddings2).sum(2)
    agg_simis = torch.mean(simis, dim=1)
    min_index = torch.argmin(agg_simis)
    dynamic_threashold[min_index][0] = 1.0
    return (simis * dynamic_threashold).mean()

def load_single_img(path:str, device):
    pil2tensor = tv.transforms.ToTensor()
    img = pil2tensor(Image.open(path)).to(device)
    return img.unsqueeze(0)

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='digital noise attack.')
    conf.add_argument("--backbone_type", type = str, default= 'ResNet' ,
                    help = "Resnet, Mobilefacenets..")
    conf.add_argument("--image_path_source", type = str, default= '/mnt/woozh/SpoofingCFAS/data/Aaron_Eckhart') 
    conf.add_argument("--image_path_target", type = str, default= '/mnt/woozh/SpoofingCFAS/data/Aaron_Peirsol')    
    args = conf.parse_args()

    imgnames_all = os.listdir(args.image_path_source)
    imgnames = imgnames_all
    print(imgnames)
    imgs = load_img_batch(imgnames, args.image_path_source, DEVICE)[:,:,:,:]
   
    
    models = get_models(DEVICE)
    model_weight = torch.Tensor([[0.1] for i in range(len(target_model))]).to(DEVICE)


    blur_func = get_gaussian_kernel().to(DEVICE)

    imgnames_t = os.listdir(args.image_path_target)
    print(imgnames_t)
    imgs_t = load_img_batch(imgnames_t, args.image_path_target , DEVICE)[:,:,:,:]
  
    large_patch_padding = 7 - 3

    embeddings1 = extract_multi_embeddings(models,imgs_t).detach()
    embeddings_ori = extract_multi_embeddings(models,imgs).detach()

    start_time = time.time()
    face_patch1 = Patch(40,82,DEVICE,lr=4/255,momentum=0.9,is_mask=True,eot=True,mask_dir='glass_eyebrow_my_design.png', opt_name="PIFGSM") #glass_3.png
   


    pil2tensor = tv.transforms.ToTensor()
    ir_region = pil2tensor(Image.open('ir_region_resize4082.png')).to(DEVICE)
    ir_region = ir_region.unsqueeze(dim=0)

    c = torch.linspace(-7, 7, 11).to(DEVICE)
    c = c.view(11, 1, 1, 1, 1).detach()

    padding = 2 #3 # 2 # + 8 
    horizontal_padding = 0 #2

    loss1_a = []
    loss2_a = []
    losses1_a = []
    loss1_ua = []
    loss2_ua = []
    losses1_ua = []


    imgs_ori = imgs.clone()

    random_pad = 4

    for i in range(1000):
        if i % 200 == 199:
            face_patch1.opt.lr = face_patch1.opt.lr / 2
        a = random.randint(-random_pad,random_pad)
        b = random.randint(-random_pad,random_pad)

        if face_patch1.opt_name == "NIFGSM":
            face_patch1.data_previous = face_patch1.data
            face_patch1.data.data = face_patch1.data_previous.data - face_patch1.opt.lr * face_patch1.opt.m * face_patch1.g_momentum
        elif face_patch1.opt_name == "PIFGSM":
            face_patch1.data_previous = face_patch1.data
            face_patch1.data.data = face_patch1.data_previous.data - face_patch1.opt.lr * face_patch1.grad_previous


        imgs1 = face_patch1.apply(imgs, (35+a-padding, 15+b-horizontal_padding)) # 17,21
        imgs1 = input_transform(imgs1, i)

        imgs2 = imgs1
        embeddings2 = extract_multi_embeddings(models, imgs2)

    
        if len(OUT_MODEL) == 0:
            embeddings_att = embeddings2
            loss1, losses1 = cross_multi_smooth_similarity_multi_targets(embeddings1,embeddings_att,False,np.array(get_threshold()), weight=model_weight)
         
        else:
            embeddings_att = embeddings2[np.delete(ALL_MODEL,OUT_MODEL),:,:]
            loss1, losses1 = cross_multi_smooth_similarity_multi_targets(embeddings1[np.delete(ALL_MODEL,OUT_MODEL),:,:],embeddings_att,False,np.array(get_threshold())[np.delete(ALL_MODEL,OUT_MODEL)], weight=model_weight)
            # embeddings_ua = embeddings2[OUT_MODEL,:,:]
            # loss1_uatt, losses1_uatt = cross_multi_smooth_similarity_multi_targets(embeddings1[OUT_MODEL,:,:],embeddings_ua,False,np.array(get_threshold())[OUT_MODEL])

        loss_tv1 = max(tv_loss(face_patch1.data), 0.04)
       
        scale = 0.1
        loss = - loss1 + scale * (loss_tv1 ) # + 1 * loss_ld1
        

        loss.backward()
        face_patch1.update(loss,dual=True)
      
        if (i % 1 == 0):
            print('epoch:{}, loss:{}'.format(i, loss.data.cpu()))
            print('epoch:{}, loss_balance:{}'.format(i, loss1))
            print('epoch:{}, loss_tvloss:{}'.format(i, loss_tv1.data.cpu()))
            # print('epoch:{}, loss_ldloss:{}'.format(i, loss_ld1.data.cpu()))
            print('epoch:{}, loss_all_models:\n{}'.format(i, losses1))
            # print('epoch:{}, loss_out_models:{}'.format(i, losses1_uatt))
        if (i % 10 == 0):
            tv.utils.save_image(imgs1[0],'./save_img_patch/test.png')          
    
   
    imgs1 = face_patch1.apply(imgs_ori, (35-padding, 15-horizontal_padding), test_mode=True)
    imgs2 = imgs1
    save_name = 'testing'
    face_patch1.savepatch('./save_img_patch/patch_{}.png'.format(save_name))
    tv.utils.save_image(imgs2[0],'./save_img_patch/image_{}.png'.format(save_name))
    
    stop_time = time.time()
    print(stop_time - start_time)

    