import os
import numpy as np
import pyiqa
import torch
from tqdm import tqdm
import pytorch_ssim
from diffusers.utils import load_image
import cv2
import torch.nn.functional as F
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import io
import matplotlib.patches as patches
import math
from SDAR import SDAR_Net
np.set_printoptions(suppress=True)
def latex_to_pil(tex_string, fontsize=20, dpi=300,t_color='white'):
    """
    将 LaTeX 字符串转换为带半透明底的 Pillow Image 对象
    """
    # 1. 创建 figure 和 axes
    # 注意：这里 figsize 设得很小，但 bbox_inches='tight' 会根据内容自动调整最终大小
    fig = plt.figure(figsize=(0.01, 0.01))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 2. 定义文本框的样式
    # facecolor: 背景颜色，'black' 表示黑色
    # alpha: 透明度，0.5 表示半透明 (0是全透明, 1是完全不透明)
    # edgecolor: 边框颜色，'none' 表示没有边框
    # pad: 内边距，文字与背景框边缘的距离，单位是点
    text_props = {
        'bbox': {
            'facecolor': 'black', 
            'alpha': 0.6, 
            'edgecolor': 'none',
            'pad': 5
        }
    }
    
    # 3. 渲染文字
    # 将定义好的 text_props 解包传入 ax.text
    # 注意：Matplotlib 的 LaTeX 需要包裹在 $ 符号中
    t = ax.text(0, 0, f"{tex_string}", fontsize=fontsize, color=t_color,fontweight='medium', **text_props)
    
    # 4. 将 figure 保存到内存缓冲区
    buf = io.BytesIO()
    # bbox_inches='tight' 确保裁剪掉多余空白，pad_inches=0 紧贴边缘
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.close(fig)
    
    buf.seek(0)
    return Image.open(buf)

def latex_to_pil_fixed_size(tex_string, fontsize=20, dpi=300, t_color='white', 
                            width_px=300, height_px=100):
    """
    将 LaTeX 字符串转换为指定【像素宽度和高度】的 Pillow Image 对象
    """
    # 1. 将像素转换为英寸 (Matplotlib 内部单位)
    width_in = width_px / dpi
    height_in = height_px / dpi
    
    # 2. 创建固定尺寸的 figure
    fig = plt.figure(figsize=(width_in, height_in))
    
    # 3. 添加坐标轴，占满整个画布 (0,0 是左下角，1,1 是右上角)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 4. 绘制背景框
    # 这里我们画一个填满整个画布的矩形
    # xy=(0,0) 是左下角，width 和 height 是画布的物理尺寸
    p = patches.FancyBboxPatch(
        (0, 0), width_in, height_in, 
        boxstyle=patches.BoxStyle("Round", pad=0), # pad=0 因为我们自己控制画布大小
        linewidth=0,
        facecolor='white',
        alpha=0.0,
        transform=ax.transData # 使用数据坐标系
    )
    ax.add_patch(p)
    
    # 5. 渲染文字 (居中显示)
    # 文字位置设为画布中心 (width_in/2, height_in/2)
    # 使用 ax.transData 确保坐标单位是英寸
    ax.text(width_in / 2, height_in / 2, tex_string, 
            fontsize=fontsize, 
            color=t_color, 
            ha='center',      # 水平居中
            va='center',      # 垂直居中
            fontweight='bold',
            transform=ax.transData)

    # 6. 设置坐标轴范围
    # 这一步是为了告诉 matplotlib 我们的画布范围是 0~width_in 和 0~height_in
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)

    # 7. 保存到内存
    buf = io.BytesIO()
    # 注意：这里去掉了 bbox_inches='tight'，以确保输出尺寸严格等于 figsize
    plt.savefig(buf, format='png', dpi=dpi, pad_inches=0.0, transparent=True)
    plt.close(fig)
    
    buf.seek(0)
    return Image.open(buf)

def get_model_pic_with_text2(output,target_256,logit=None):
    output=output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    test_s=output
    test_s=Image.fromarray(test_s.astype(np.uint8))
    # draw = ImageDraw.Draw(test_s)
    psnr256=compute_psnr(output,target_256)
    text = "%.2f dB"%(psnr256)

    text_psnr=latex_to_pil_fixed_size(text,fontsize=8.2,width_px=256,height_px=35,t_color="black")
    # 获取文字尺寸
    text_width, text_height = text_psnr.size
    # 计算文字位置（居中于图片底部）
    width, height = test_s.size
    x = (width - text_width) // 2
    y = height #- text_height  # 在图片底部留一些边距
    new_image = Image.new('RGBA', (width, height + text_height), (255, 255, 255, 1))
    new_image.paste(test_s,(0,0))
    new_image.paste(text_psnr,(x,y),text_psnr)
    if logit!=None:
        weights=F.softmax(logits,dim=-1)[0]
        text_w=f""
        for i in range(num_b-1):
            text_w+=f"$w_{i}$: {weights[i]:.2f}\n"
        text_w+=f"$w_{num_b-1}$: {weights[num_b-1]:.2f}"
        w_img=latex_to_pil(text_w,fontsize=7.5)
        text_width, text_height = w_img.size
        new_image.paste(w_img,(width-text_width,0),w_img)
    return np.array(new_image)

def get_model_pic(output):
    test_s=output
    test_s=Image.fromarray(test_s.astype(np.uint8))
    width, height = test_s.size
    new_image = Image.new('RGBA', (width, height + 35), (255, 255, 255, 1))
    new_image.paste(test_s,(0,0))
    return np.array(new_image)
MUSIQ = pyiqa.create_metric('musiq-spaq', as_loss=True).cuda()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)

def compute_psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

UIQM=pytorch_ssim.calculate_uiqm
UCIQE=pytorch_ssim.calculate_uciqe
SSIM=pytorch_ssim.SSIM().float().cuda()
font_path = 'arial.ttf'
font_size = 24
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default(size=38)
    font1 = ImageFont.load_default(size=36)
num_b=3
model_name="SDAR"

generator=SDAR_Net(num_branch=num_b).cuda()
generator.load_state_dict(torch.load("./pre_trains/UIEB/ckpt-best.pth"))
generator.eval()
save_path="./test_outs_"
path='/data/UnderwaterDatasets/UIEB-new/test/'
path_list = os.listdir(path+"input/")
path_list.sort()
i=1
PSNR256=[]
PSNR256_0=[]
PSNR256_1=[]
PSNR256_2=[]
PSNRfull=[]
SSIM256=[]
SSIM256_2=[]
SSIMfull=[]
LUM256=[]
LUMfull=[]
CON256=[]
CONfull=[]
STU256=[]
STUfull=[]
UIQMfull=[]
UCIQEfull=[]
UIQM256=[]
UIQM256_2=[]
UCIQE256=[]
MUSIQ256=[]
MUSIQ256_0=[]
MUSIQ256_1=[]
MUSIQ256_2=[]
weights_lists=[]
PSNRs_lists=[]
UIQMs_lists=[]
UCIQEs_lists=[]
MUSIQs_lists=[]
image_metrics=[]
for i in range(num_b):
    weights_list=[]
    weights_lists.append(weights_list)
    PSNRs_list=[]
    PSNRs_lists.append(PSNRs_list)
    UIQMs_list=[]
    UIQMs_lists.append(UIQMs_list)
    UCIQEs_list=[]
    UCIQEs_lists.append(UCIQEs_list)
    MUSIQs_list=[]
    MUSIQs_lists.append(MUSIQs_list)

if not os.path.exists(save_path+model_name):
    os.makedirs(save_path+model_name)
for item in tqdm(path_list):
    org_input = load_image(path+"input/"+item)
    org_input=np.array(org_input)
    org_target = load_image(path+"target/"+item)
    org_target=np.array(org_target)

    input=cv2.resize(org_input,(256,256))
    input_np = np.array(input).astype(dtype)
    input= torch.from_numpy(input_np/255).permute(2,0,1).unsqueeze(0).cuda()

    target=cv2.resize(org_target,(256,256))
    target = np.array(target).astype(dtype)
    target= torch.from_numpy(target/255).permute(2,0,1).unsqueeze(0).cuda()
    output,logits, output_list = generator.forward_route(input,return_logits=True,return_proc_outs=True)
    output0=output_list[0]
    output1=output_list[1]
    output2=output_list[2]
    weights=F.softmax(logits,dim=-1)[0]
    target_256=np.array(cv2.resize(org_target,(256,256))).astype(dtype)
    for i in range(num_b):
        weights_lists[i].append(weights[i].detach().cpu().numpy())
        output_i=output_list[i]
        output_i=output_i[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        psnr_i=compute_psnr(output_i,target_256)
        PSNRs_lists[i].append(psnr_i)
    output_st=get_model_pic_with_text2(output,target_256,logit=logits)
    output0_st=get_model_pic_with_text2(output0,target_256)
    output1_st=get_model_pic_with_text2(output1,target_256)
    output2_st=get_model_pic_with_text2(output2,target_256)
    musiq256 = MUSIQ(output).detach().cpu().numpy()
    MUSIQ256.append(musiq256)
    musiq256_0 = MUSIQ(output0).detach().cpu().numpy()
    MUSIQ256_0.append(musiq256_0)
    musiq256_1 = MUSIQ(output1).detach().cpu().numpy()
    MUSIQ256_1.append(musiq256_1)
    musiq256_2 = MUSIQ(output2).detach().cpu().numpy()
    MUSIQ256_2.append(musiq256_2)
    ssim256=SSIM(output,torch.from_numpy(target_256/255).permute(2,0,1).unsqueeze(0).cuda())
    ssim256_2=SSIM(output2,input)

    output=output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    psnr256=compute_psnr(output,target_256)
    PSNR256.append(psnr256)

    output0=output0[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    psnr256_0=compute_psnr(output0,target_256)
    PSNR256_0.append(psnr256_0)
    output1=output1[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    psnr256_1=compute_psnr(output1,target_256)
    PSNR256_1.append(psnr256_1)
    output2=output2[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    psnr256_2=compute_psnr(output2,target_256)
    PSNR256_2.append(psnr256_2)
 
    SSIM256.append(ssim256.cpu().detach().numpy())

    uiqm_256=UIQM(output)
    UIQM256.append(uiqm_256)
    uciqe_256=UCIQE(output)
    UCIQE256.append(uciqe_256)
    input_np=get_model_pic(input_np)
    target_256=get_model_pic(target_256)
    test_s_2=np.concatenate([input_np,output0_st,output1_st,output2_st,output_st,target_256],axis=1)#input_np,
    test_s_2=Image.fromarray(test_s_2.astype(np.uint8))
    test_s_2.save(save_path+model_name+"/"+item[:-3]+"png")
PSNR256=np.array(PSNR256)    
print("PSNR-256: %.4f"%(PSNR256.mean()))

SSIM256=np.array(SSIM256)    
print("SSIM-256: %.4f"%(SSIM256.mean()))

UIQM256=np.array(UIQM256)    
print("UIQM-256: %.4f"%(UIQM256.mean()))

UCIQE256=np.array(UCIQE256)    
print("UCIQE-256: %.4f"%(UCIQE256.mean()))

MUSIQ256=np.array(MUSIQ256)    
print("MUSIQ-256: %.4f"%(MUSIQ256.mean()))

weights_lists=np.array(weights_lists)
for i in range(num_b):
    print("avg_W_"+str(i)+": %.4f"%(weights_lists[i].mean()))

PSNR256_0=np.array(PSNR256_0)    
print("PSNR-256-0: %.4f"%(PSNR256_0.mean()))
MUSIQ256_0=np.array(MUSIQ256_0)    
print("MUSIQ-256-0: %.4f"%(MUSIQ256_0.mean()))

PSNR256_1=np.array(PSNR256_1)    
print("PSNR-256-1: %.4f"%(PSNR256_1.mean()))
MUSIQ256_1=np.array(MUSIQ256_1)    
print("MUSIQ-256-1: %.4f"%(MUSIQ256_1.mean()))

PSNR256_2=np.array(PSNR256_2)    
print("PSNR-256-2: %.4f"%(PSNR256_2.mean()))
MUSIQ256_2=np.array(MUSIQ256_2)    
print("MUSIQ-256-2: %.4f"%(MUSIQ256_2.mean()))
