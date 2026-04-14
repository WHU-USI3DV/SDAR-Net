import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2
from skimage import transform
from scipy import ndimage
import lpips
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def calculate_uciqe(img):
    img_bgr =img

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    # if nargin == 1:                                 # According to training result mentioned in the paper:
    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val
    # image = img
    # # image = cv2.imread(image)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    # H, S, V = cv2.split(hsv)
    # delta = np.std(H) / 180
    # # 色度的标准差
    # mu = np.mean(S) / 255  # 饱和度的平均值
    # # 求亮度对比值
    # n, m = np.shape(V)
    # number = math.floor(n * m / 100)
    # v = V.flatten() / 255
    # v.sort()
    # bottom = np.sum(v[:number]) / number
    # v = -v
    # v.sort()
    # v = -v
    # top = np.sum(v[:number]) / number
    # conl = top - bottom
    # uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    # return uciqe


def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    RG = R - G
    YB = (R+G)/2 -B
    K = R.shape[0]*R.shape[1]
    RG1 = RG.reshape(1,K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanRG = np.sum(RG1)/N
    deltaRG = np.sqrt(np.sum((RG1-meanRG)**2)/N)

    YB1 = YB.reshape(1,K)
    YB1 = np.sort(YB1)
    alphaL = 0.1
    alphaR = 0.1
    YB1 = YB1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB)**2)/N)
    uicm = -0.0268*np.sqrt(meanRG**2+meanYB**2)+ 0.1586*np.sqrt(deltaYB**2+deltaRG**2)
    return uicm

def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m-m%patchez+patchez)
        y = int(n-n%patchez+patchez)
        R = transform.resize(R,(x,y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))
    m = R.shape[0]
    n = R.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    AMEER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = R[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEER = 1/(k1*k2) *np.abs(AMEER)
    AMEEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = G[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEG = 1/(k1*k2) *np.abs(AMEEG)
    AMEEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = B[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEB = 1/(k1*k2) *np.abs(AMEEB)
    uiconm = AMEER +AMEEG +AMEEB
    return uiconm

def _uism(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    hx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    hy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest')+ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest')+ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest')+ndimage.convolve(B, hy, mode='nearest'))
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))
    m = SobelR.shape[0]
    n = SobelR.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    EMER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelR[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max/Min)
    EMER = 2/(k1*k2)*np.abs(EMER)

    EMEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelG[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max/Min)
    EMEG = 2/(k1*k2)*np.abs(EMEG)
    EMEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelB[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max/Min)
    EMEB = 2/(k1*k2)*np.abs(EMEB)
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism

def calculate_uiqm(img):
    x = img
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm