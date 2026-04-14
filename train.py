import math
import os
import numpy as np
import pyiqa
import torch
import torch.nn as nn
from SDAR import SDAR_Net
from losses import MyLoss,VGG19_PercepLoss
from modules import GetGradientNopadding
import pytorch_ssim 
from torch.autograd import Variable
from PIL import Image,ImageDraw, ImageFont
from torch.optim import lr_scheduler
import cv2
from train_utils import get_data
import torch.nn.functional as F
import time as time
import datetime
from datetime import datetime as dt
from tqdm import tqdm
from adamp import AdamP
from diffusers.utils import load_image
font = ImageFont.load_default(size=36)
def get_model_pic_with_text(U_shape_output,target_256,logit=None):
    U_shape_output=U_shape_output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    test_s=U_shape_output
    test_s=Image.fromarray(test_s.astype(np.uint8))
    draw = ImageDraw.Draw(test_s)
    psnr256=compute_psnr(U_shape_output,target_256)
    text = "PSNR: %.4f"%(psnr256)

    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 计算文字位置（居中于图片底部）
    width, height = test_s.size
    x = (width - text_width) // 2
    y = height - text_height - 10  # 在图片底部留一些边距

    # 绘制文字
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    if logit!=None:
        weights=F.softmax(logit,dim=-1)[0]
        text_w=""
        for i in range(num_b):
            text_w+="  W_"+str(i)+": %.4f"%(weights[i])+"\n"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = width - text_width - 2
        y = 2
        draw.text((x, y), text_w, fill=(255, 255, 255), font=font)
    return np.array(test_s)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

UIQM=pytorch_ssim.calculate_uiqm
UCIQE=pytorch_ssim.calculate_uciqe
MUSIQ = pyiqa.create_metric('musiq-spaq', as_loss=True).cuda()

def compute_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.unsqueeze(-1)  # [B, 1]

def test(generator,pre=False):
    if pre:
        generator.load_state_dict(torch.load(DIR_CHECKPOINTS+"/'ckpt-best.pth'"))

    generator.eval()

    path='/data/UnderwaterDatasets/'+dataset_name+'-new/test/'#要改

    path_list = os.listdir(path+"input/")
    path_list.sort()
    PSNR=[]
    PSNR2=[]
    SSIM256=[]
    UIQM256=[]
    UCIQE256=[]
    MUSIQ256=[]
    for item in tqdm(path_list):
        org_input = load_image(
            path+"input/"+item
        )
        org_input=np.array(org_input)
        org_target = load_image(
            path+"target/"+item
        )
        org_target=np.array(org_target)

        U_shape_input=cv2.resize(org_input,(256,256))
        U_shape_input_np = np.array(U_shape_input).astype(dtype)
        U_shape_input= torch.from_numpy(U_shape_input_np/255).permute(2,0,1).unsqueeze(0).cuda()

        U_shape_output,logits, output_list = generator.forward_route(U_shape_input,return_logits=True,return_proc_outs=True)
        U_shape_output2=output_list[1]
        target_256=np.array(cv2.resize(org_target,(256,256))).astype(dtype)
        musiq256 = MUSIQ(U_shape_output).detach().cpu().numpy()
        MUSIQ256.append(musiq256)
        ssim256=SSIM(U_shape_output,torch.from_numpy(target_256/255).permute(2,0,1).unsqueeze(0).cuda())
        ssim256=ssim256.detach().cpu().numpy()
        U_shape_output_np=U_shape_output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        psnr256=compute_psnr(U_shape_output_np,target_256)
        PSNR.append(psnr256)
        SSIM256.append(ssim256) 
        uiqm_256=UIQM(U_shape_output_np)
        UIQM256.append(uiqm_256)
        uciqe_256=UCIQE(U_shape_output_np)
        UCIQE256.append(uciqe_256)

        U_shape_output2_np=U_shape_output2[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        psnr2=compute_psnr(U_shape_output2_np,target_256)
        PSNR2.append(psnr2)
        U_shape_output_s=get_model_pic_with_text(U_shape_output,target_256,logit=logits)
        U_shape_output2_s=get_model_pic_with_text(U_shape_output2,target_256)
        test_s=np.concatenate([U_shape_input_np,target_256,U_shape_output2_s,U_shape_output_s],axis=1)
        test_s=Image.fromarray(test_s.astype(np.uint8))
        test_s.save(save_path+model_name+"/"+item) 

    PSNR=np.array(PSNR)
    SSIM256=np.array(SSIM256)         
    UIQM256=np.array(UIQM256)
    UCIQE256=np.array(UCIQE256)
    MUSIQ256=np.array(MUSIQ256)

    PSNR2=np.array(PSNR2)
    print("   PSNR-256: %.4f    SSIM-256: %.4f"%(PSNR.mean(),SSIM256.mean()))
    print("   UIQM-256: %.4f    UCIQE-256: %.4f   MUSIQ-256: %.4f"%(UIQM256.mean(),UCIQE256.mean(),MUSIQ256.mean()))
    print("   PSNR2: %.4f"%(PSNR2.mean()))
    return PSNR.mean(),PSNR2.mean(),UIQM256.mean()

def recon_losses(output,GT):
    perpetual_loss = L_vgg(output, GT)
    structure_loss = loss_str(output, GT)
    gradient_loss = loss_grad(get_grad(output,gray=False), get_grad(GT,gray=False))
    loss_final=structure_loss+perpetual_loss*0.3+gradient_loss*0.1
    return loss_final

def select_best(output_list, gt,max_idx):

    B = gt.shape[0]

    # Step 2: 找出需要训练的样本（伪标签 ≠ 0）
    valid_mask = (max_idx != 0)                 # [B], bool
    num_valid = valid_mask.sum().item()

    if num_valid == 0:
        # 返回空张量（shape 一致），或 None
        # 这里返回两个空张量，shape 为 [0, C, H, W]
        # empty = torch.empty(0, *gt.shape[1:], device=gt.device)
        return None,None

    # Step 3: 对 valid 样本，选择对应的 output
    # 堆叠 outputs: [B, 3, C, H, W]
    outputs_cat = torch.stack(output_list, dim=1)  # [B, 3, C, H, W]

    # 只处理 valid 样本
    # valid_indices = torch.where(valid_mask)[0]     # [num_valid]
    valid_max_idx = max_idx[valid_mask]           # [num_valid]

    # gather 对应输出
    idx_expanded = valid_max_idx.view(-1, 1, 1, 1, 1).expand(
        -1, 1, *gt.shape[1:]
    )  # [num_valid, 1, C, H, W]

    valid_outputs = outputs_cat[valid_mask]       # [num_valid, 3, C, H, W]
    best_output = torch.gather(valid_outputs, dim=1, index=idx_expanded).squeeze(1)  # [num_valid, C, H, W]
    gt_valid = gt[valid_mask]                     # [num_valid, C, H, W]

    return best_output, gt_valid

dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)
save_dir="./project_logs/UIE"
dataset_name="UIEB"
bs=2
train_loader=get_data(batch_size=bs,path="/data/UnderwaterDatasets/UIEB-new/train/", img_size=256)
loader=train_loader
MSE = nn.MSELoss(size_average=True).cuda()
SSIM = pytorch_ssim.SSIM().cuda()
loss_grad = nn.L1Loss().cuda()
loss_str = MyLoss().cuda()
get_grad = GetGradientNopadding().cuda()

num_b=3
model_name="SDAR"
model=SDAR_Net(num_branch=num_b).cuda()
LR=0.0001
checkpoint_interval=20
n_epochs=200
route_ep=50
# Optimizers
G_params = []
router_params = []

for name, param in model.named_parameters():
    if 'adaptive_route' in name:
        router_params.append(param)
    else:
        G_params.append(param)
print('Route Parameters: ', sum(p.numel() for p in router_params))
optimizer_G = AdamP(G_params, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler_G=lr_scheduler.MultiStepLR(optimizer_G, milestones=[25,50,75,100], gamma=0.5)
optimizer_router = AdamP(router_params, lr=2e-5, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler_router=lr_scheduler.MultiStepLR(optimizer_router, milestones=[25,50,75,100], gamma=0.3)

L_vgg = VGG19_PercepLoss().cuda()
loss_grad = nn.L1Loss().cuda()
loss_str = MyLoss().cuda()
get_grad = GetGradientNopadding().cuda()
loss_fs=recon_losses
lambda_style=10
lambda_route=1

use_pretrain=False
output_dir = os.path.join(save_dir, model_name,dataset_name,'%s', dt.now().strftime('%y-%m-%d-%H-%M-%S'))
DIR_CHECKPOINTS = output_dir % 'checkpoints'
DIR_LOGS = output_dir % 'logs'
save_path="./train_outs_"
if not os.path.exists(save_path+model_name):
    os.makedirs(save_path+model_name)
if not os.path.exists(DIR_CHECKPOINTS):
    os.makedirs(DIR_CHECKPOINTS)
if not os.path.exists(DIR_LOGS):
    os.makedirs(DIR_LOGS)
if use_pretrain:
    # Load pretrained models
    start_epoch=40
    model.load_state_dict(torch.load("./pre_trains/UIEB/ckpt-best.pth"),strict=False)
    print('successfully loading epoch {}'.format(start_epoch))
else:
    start_epoch = 0
    print('No pretrain model found, training will start from scratch！')

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
epochs=start_epoch
print('Parameters: ', sum(p.numel() for p in model.parameters()))
# ingnored when opt.mode=='S'
psnr_list = [] 
prev_time = time.time()
best=0
best_epoch=0
curiter=1
weights_r=torch.ones(size=[bs])
for epoch in range(epochs,n_epochs):
    with tqdm(loader) as t:
        for i, batch in enumerate(t):

            # Model inputs
            Input = Variable(batch[0]).cuda() 
            GT = Variable(batch[1]).cuda()
            optimizer_G.zero_grad()

            output_in = model.forward_recon(Input)
            loss_final_in=loss_fs(output_in,Input)
            output_gt = model.forward_recon(GT)
            loss_final_gt=loss_fs(output_gt,GT)
            loss_final=loss_final_in+loss_final_gt
            loss_final.backward(retain_graph=True)

            output,style_loss = model.forward_style_loss(Input,GT)
            loss_final=loss_fs(output,GT)
            loss_final=loss_final+style_loss*lambda_style
            loss_final.backward(retain_graph=True)

            if epoch>=route_ep:
                with torch.no_grad():
                    output_r=Input
                    psnr_0t=compute_psnr_batch(output_r,GT)
                    psnrs=[psnr_0t]
                    for i in range(num_b-1):
                        output_r=model.forward(output_r)
                        psnr_nt=compute_psnr_batch(output_r,GT)
                        psnrs.append(psnr_nt)
                    psnr_stack=torch.cat(psnrs,dim=1)
                    max_idx=torch.argmax(psnr_stack,dim=1)
                    weights_r=max_idx.detach()
                optimizer_router.zero_grad()
                output,logits,style_loss, output_list = model.forward_route_style_loss(Input,GT,return_logits=True,return_proc_outs=True)
                best_outputs,valid_gt=select_best(output_list,GT,max_idx)
                loss_router=F.cross_entropy(logits, weights_r,size_average=True)
                loss_final=loss_fs(output,GT)
                loss_final=loss_final+loss_router*lambda_route+style_loss*lambda_style
                if best_outputs!=None:
                    loss_p_outs=loss_fs(best_outputs,valid_gt)
                    loss_final=loss_final+loss_p_outs
                loss_final.backward(retain_graph=True)
                optimizer_router.step()
            optimizer_G.step()
            

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(loader) + i
            batches_left = n_epochs * len(loader) - batches_done
            out_train= torch.clamp(output, 0., 1.) 
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch,
                                                    n_epochs,
                                                    i,
                                                    len(loader),))
 
            if epoch>=route_ep:
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [loss_final.item(),loss_router.item(), style_loss.item(),max_idx.float().mean().item()]])
            else:
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [loss_final.item()]])
    scheduler_G.step()
    test_psnr,test_psnr2,test_musiq=test(model)
    if epoch>route_ep:
        test_best=test_psnr
    elif epoch==route_ep:
        # best=0
        test_best=test_psnr
    else:
        test_best=test_psnr
    if test_best>best:
        best=test_best
        best_epoch=epoch
        file_name = 'ckpt-best.pth'
        output_path = os.path.join(DIR_CHECKPOINTS, file_name)
        torch.save(model.state_dict(), output_path)
        print('Saved checkpoint to %s ...' % output_path)
    elif checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        file_name = 'ckpt-epoch-%05d.pth' % epoch
        output_path = os.path.join(DIR_CHECKPOINTS, file_name)
        torch.save(model.state_dict(), output_path)
        print('Saved checkpoint to %s ...' % output_path)
    print('Best Performance: Epoch %d --  %.4f   current: %.4f' % (best_epoch,best,test_best))
 