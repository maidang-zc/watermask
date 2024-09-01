from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np

# torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_ssim as ssim
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
def is_dic(x):
    return type(x) == type([])



def tensor2np(x, isMask=False):
    if isMask:
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x = ((x.cpu().detach()))*255
    else:
        x = x.cpu().detach()
        mean = 0
        std = 1
        x = (x * std + mean)*255
		
    return x.numpy().transpose(0,2,3,1).astype(np.uint8)

def save_output(inputs, preds, save_dir, img_fn, extra_infos=None,  verbose=False, alpha=0.5):
    outs = []
    image, bg_gt,mask_gt = inputs['I'], inputs['bg'], inputs['mask']
    image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)
    # fg_gt = cv2.cvtColor(tensor2np(fg_gt)[0], cv2.COLOR_RGB2BGR)
    bg_gt = cv2.cvtColor(tensor2np(bg_gt)[0], cv2.COLOR_RGB2BGR)
    mask_gt = tensor2np(mask_gt, isMask=True)[0]

    bg_pred,mask_preds = preds['bg'], preds['mask']
    # fg_pred = cv2.cvtColor(tensor2np(fg_pred)[0], cv2.COLOR_RGB2BGR)
    bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
    mask_preds = [tensor2np(m, isMask=True)[0] for m in mask_preds]
    # main_mask = mask_preds[-2]
    mask_pred = mask_preds[0]
    outs = [image, bg_gt, bg_pred, mask_gt, mask_pred] #, main_mask]
    outimg = np.concatenate(outs, axis=1)
	
    if verbose==True:
        # print("show")
        cv2.imshow("out",outimg)
        cv2.waitKey(0)
    else:
        psnr = extra_infos['psnr']
        rmsew = extra_infos['rmsew']
        f1 = extra_infos['f1']

        img_fn = os.path.split(img_fn)[-1]
        out_fn = os.path.join(save_dir, "{}_psnr_{:.2f}_rmsew_{:.2f}_f1_{:.4f}{}".format(os.path.splitext(img_fn)[0],psnr,rmsew, f1, os.path.splitext(img_fn)[1]))
        cv2.imwrite(out_fn, outimg)

def save_outputx(inputs, preds, save_dir, index= 0):
    outs = []
    image, bg_gt,mask_gt = inputs['I'], inputs['bg'], inputs['mask']
    image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)
    # bg_gt = cv2.cvtColor(tensor2np(bg_gt)[0], cv2.COLOR_RGB2BGR)
    # mask_gt = tensor2np(mask_gt, isMask=True)[0]

    bg_pred,mask_preds = preds['bg'], preds['mask']
    # fg_pred = cv2.cvtColor(tensor2np(fg_pred)[0], cv2.COLOR_RGB2BGR)
    bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
    mask_preds = [tensor2np(m, isMask=True)[0] for m in mask_preds]
    # main_mask = mask_preds[-2]
    mask_pred = mask_preds[0]
    outs = [image, bg_pred, mask_pred] #, main_mask]
    outimg = np.concatenate(outs, axis=1)
    # img_fn = os.path.split(img_fn)[-1]
    out_fn = os.path.join(save_dir,str(index)+'.jpg')
    cv2.imwrite(out_fn, outimg)


import lpips

def mynorm(input):
    max_val = torch.max(input)
    min_val = torch.min(input)
    input = (input-min_val)/(max_val-min_val)
    input = 2*input - 1
    return input



def main(args):
    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    elif args.dataset == 'logo':
        dataset_func = datasets.LOGODataset
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    data_loaders = (None,val_loader)

    Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

    
    model = Machine
    model.model.eval()
    print("==> testing VM model ")
    rmses = AverageMeter()
    rmsews = AverageMeter()
    ssimesx = AverageMeter()
    psnresx = AverageMeter()
    maskIoU = AverageMeter()
    maskF1 = AverageMeter()
    prime_maskIoU = AverageMeter()
    prime_maskF1 = AverageMeter()
    processTime = AverageMeter()

    prediction_dir = os.path.join(args.checkpoint,'rstbest')
    if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
    
    save_flag = True # 保存标志
    with torch.no_grad():
        lpips_lst = []
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        for i, batches in enumerate(model.val_loader):

            inputs = batches['image'].to(model.device)
            target = batches['target'].to(model.device)
            mask =batches['mask'].to(model.device)
            img_path = batches['img_path']

            # select the outputs by the giving arch
            start_time = time.time()
            outputs = model.model(model.norm(inputs))
            process_time = time.time() - start_time
            processTime.update((process_time*1000), inputs.size(0))

            imoutput,immask_all,_,_ = outputs
            imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            
            immask = immask_all[0]

            imfinal =imoutput*immask + model.norm(inputs)*(1-immask)

            im_lpip = mynorm(imfinal)
            tar_lpip = mynorm(target)
            lpip = loss_fn_alex(im_lpip, tar_lpip).item()

            psnrx = 10 * log10(1 / F.mse_loss(imfinal,target).item())       
            final_np = (imfinal.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            target_np = (target.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            # ssimx = ssim(final_np, target_np, multichannel=True)
            ssimx = pytorch_ssim.ssim(imfinal, target)
            
            
            
            rmsex = compute_RMSE(imfinal, target, mask, is_w=False)
            rmsewx = compute_RMSE(imfinal, target, mask, is_w=True)
            rmses.update(rmsex, inputs.size(0))
            rmsews.update(rmsewx, inputs.size(0))
            psnresx.update(psnrx, inputs.size(0))
            ssimesx.update(ssimx, inputs.size(0))

            lpips_lst.append(lpip)


            # main_mask = immask_all[1::2]
            # comp_mask = immask_all[2::2]
            out_mask = immask_all[0]
            comp_mask = immask_all[0]
            
            comp_sets = []
            prime_mask_pred = torch.where(out_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
            mask_pred = torch.where(comp_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
           
            iou = compute_IoU(prime_mask_pred, mask)
            prime_maskIoU.update(iou)
            f1 = FScore(prime_mask_pred, mask).item()
            prime_maskF1.update(f1, inputs.size(0))

            iou = compute_IoU(mask_pred, mask)
            maskIoU.update(iou)
            f1 = FScore(mask_pred, mask).item()
            maskF1.update(f1, inputs.size(0))

            if save_flag:
                save_output(
                    inputs={'I':inputs, 'bg':target,  'mask':mask}, 
                    preds={'bg':imfinal, 'mask':immask_all}, 
                    save_dir=prediction_dir, 
                    img_fn=img_path[0], 
                    extra_infos={"psnr":psnrx, "rmsew":rmsewx, "f1":f1},
                    verbose=False
                )
            if i % 100 == 0:
                print("Batch[%d/%d]| PSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | lpips:%.6f | time:%.2f"
                %(i,len(model.val_loader),psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg,sum(lpips_lst)/len(lpips_lst), processTime.avg))
    print("Total:\nPSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | lpips:%.6f | time:%.2f"
                %(psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, sum(lpips_lst)/len(lpips_lst), processTime.avg))
    print("DONE.\n")


from PIL import Image
from torchvision import transforms
def evalute(args):
    for i in range(1, 4):

        imagepath = f'syydataset/{i}.png'

        # 创建一个图像到tensor变换 
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 创建一个 PIL 图像
        img = Image.open(imagepath) # channels, height, width 如果是4通道就只取前3个通道
        # 将图像转换为张量
        img_tensor = transform(img)[:3]
        # img_tensor = img_tensor.permute(2, 0, 1)
        input_tensor = img_tensor.unsqueeze(0) # 添加一个维度，因为模型需要一个批次维度 
        print(input_tensor.shape) # torch.Size([1, 3, 256, 256])
        # 创建模型
        Machine = models.__dict__[args.models](args=args)
        model = Machine
        model.model.eval()
        input_tensor = input_tensor.to(model.device)
        print("==> testing VM model ")
        
        # 保存路径
        prediction_dir = os.path.join(args.checkpoint,'rsttest')
        if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
        # 模型输出
        outputs = model.model(model.norm(input_tensor))
        
        imoutput,immask_all,_,_ = outputs
        imoutput = imoutput[0] if is_dic(imoutput) else imoutput
        
        immask = immask_all[0]

        imfinal =imoutput*immask + model.norm(input_tensor)*(1-immask)

        save_outputx(
            inputs={'I':input_tensor, 'bg':None,  'mask':None}, 
            preds={'bg':imfinal, 'mask':immask_all}, 
            save_dir=prediction_dir,  
            index = i
        )

if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    evalute(parser.parse_args())
    
