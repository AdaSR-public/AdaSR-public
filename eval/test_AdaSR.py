import argparse

import os
import glob
import shutil

from PIL import Image

import pandas as pd
import numpy as np

import torch
import torchvision.transforms.functional as TF

from tqdm import tqdm

from src.dnn.models import build_sr, build_qpnet
from src.enc.jpeg_adasr import JpegAdaSR
from src.dnn.qp_module import QNModule

from eval.utility import sr_inference, calc_psnr, tensor2pil

model_to_mac = {}
model_to_mac['fsrcnn'] = {16: 138344.0, 36: 297984.0, 48: 393768.0, 56: 457624.0}
model_to_mac['carn'] ={36: 389688.0, 40: 474736.0, 48: 669888.0, 52: 779992.0, 64: 1160416.0}
model_to_mac['edsr'] = {32: 715680.0, 40: 1107696.0, 48: 1584960.0, 56: 2147472.0, 64: 2795232.0}

# jpeg header size (in bytes)
JPEG_HEADER_SIZE = 625

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='./dataset/test4k')
    parser.add_argument('--save_image', action='store_true')
    
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num_importance', type=int, default=10)
    
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--split_by', type=int, default=1)
    
    parser.add_argument('--quality', type=int, default=65)
    
    parser.add_argument('--profiles', type=str, default='./dataset/profile/size_qual_stats.csv')
    
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='./result/')
    
    parser.add_argument('--sr_model_name', type=str, default='edsr')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--sr_checkpoint_dir', type=str, default='./pretrained')
    
    args = parser.parse_args()
    
    # set torch device
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    
    # load test dataset
    if args.data_dir.find('DIV2K') != -1:
        data_type = 'DIV2K'
        lr_data_dir = os.path.join(args.data_dir, f'DIV2K_valid_LR_bicubic/X{args.scale}')
        target_data_dir = os.path.join(args.data_dir, 'DIV2K_valid_HR/')
        split_by = args.split_by
            
        lr_image_list = sorted(glob.glob(os.path.join(lr_data_dir, '*')))
        target_image_list = sorted(glob.glob(os.path.join(target_data_dir, '*')))

    elif args.data_dir.find('test8k') != -1:
        data_type = 'test8k'
        lr_data_dir = os.path.join(args.data_dir, f'LR/X{args.scale}')
        target_data_dir = os.path.join(args.data_dir, f'HR/X{args.scale}')
        split_by = args.split_by
        
        lr_image_list = sorted(glob.glob(os.path.join(lr_data_dir, '*')))
        target_image_list = sorted(glob.glob(os.path.join(target_data_dir, '*')))
        
    elif args.data_dir.find('test4k') != -1:
        data_type = 'test4k'
        lr_data_dir = os.path.join(args.data_dir, f'LR/X{args.scale}')
        target_data_dir = os.path.join(args.data_dir, f'HR/X{args.scale}')
        split_by = args.split_by

        lr_image_list = sorted(glob.glob(os.path.join(lr_data_dir, '*')))
        target_image_list = sorted(glob.glob(os.path.join(target_data_dir, '*')))
    
    elif args.data_dir.find('test2k') != -1:
        data_type = 'test2k'
        lr_data_dir = os.path.join(args.data_dir, f'LR/X{args.scale}')
        target_data_dir = os.path.join(args.data_dir, f'HR/X{args.scale}')
        split_by = args.split_by

        lr_image_list = sorted(glob.glob(os.path.join(lr_data_dir, '*')))
        target_image_list = sorted(glob.glob(os.path.join(target_data_dir, '*')))

    else:
        raise Exception("Unsupported dataset")
    
    # set result path
    result_dir = os.path.join(args.result_dir, data_type, args.sr_model_name, f'X{args.scale}', f'{args.quality}')
    
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    else:
        os.makedirs(result_dir)
        
    log_dir = os.path.join(result_dir, 'log')
    visual_result_dir = os.path.join(result_dir, 'visual')
    
    os.makedirs(log_dir)
    os.makedirs(visual_result_dir)
    
    log_path = os.path.join(log_dir, 'log.txt')
    
    # temporal space for saving image file
    tmp_space = './tmp_space.jpg'
    
    # SR channel list for adjusting computation
    sr_channel_list = list(model_to_mac[args.sr_model_name].keys())
    
    # load SR network list
    sr_models = []
    for num_channels in sr_channel_list:
        if args.sr_model_name == 'edsr':
            num_blocks = 8
            model = build_sr(args.sr_model_name, 3, 3, num_channels, num_blocks, args.scale)
            checkpoint_path = os.path.join(args.sr_checkpoint_dir, f'EDSR_B{num_blocks}_F{num_channels}_S{args.scale}.pth')
        elif args.sr_model_name == 'fsrcnn':
            num_blocks = 4
            model = build_sr(args.sr_model_name, 3, 3, num_channels, num_blocks, args.scale)
            checkpoint_path = os.path.join(args.sr_checkpoint_dir, f'FSRCNN_B{num_blocks}_F{num_channels}_S{args.scale}.pth')
        elif args.sr_model_name == 'carn':
            num_blocks = 6
            model = build_sr(args.sr_model_name, 3, 3, num_channels, num_blocks, args.scale)
            checkpoint_path = os.path.join(args.sr_checkpoint_dir, f'CARN_B{num_blocks}_F{num_channels}_S{args.scale}.pth')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model.to(device)
        sr_models += [model]
    
    # load QP-NET
    num_channels = 128
    num_layers = 2
    output_dim = 10
    qpnet = build_qpnet(num_layers, num_channels, output_dim, args.patch_size)
    checkpoint_path = f'{args.checkpoint_dir}/QPNET_L{num_layers}_F{num_channels}_O{output_dim}_{args.patch_size}.pt'

    qpnet.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    qpnet.to(device)
    qpnet.eval()
    
    # Encoding Emulator
    jpeg_adasr = JpegAdaSR(args.profiles, args.num_importance)
    
    # QP-NET Module
    qn_module = QNModule(qpnet, args.patch_size, device)
    
    # test process
    avg_base_psnr = np.zeros(len(sr_channel_list))
    avg_base_computing = np.zeros(len(sr_channel_list))
    avg_base_bpp = 0.
    
    avg_ada_psnr = np.zeros(len(sr_channel_list))
    avg_ada_computing = np.zeros(len(sr_channel_list))
    avg_ada_bpp = 0.
    
    for index, (lr_image_path, hr_image_path) in enumerate((tqdm(zip(lr_image_list, target_image_list), total=len(lr_image_list), desc=f"Evaluating on {data_type}_x{args.scale}"))):
        base_name = os.path.basename(lr_image_path)
        # load origin raw image
        origin_image = Image.open(lr_image_path)
        origin_image.load()
        
        # crop image divided by patch size
        cropped_width = (origin_image.width//args.patch_size)*args.patch_size
        cropped_height = (origin_image.height//args.patch_size)*args.patch_size
        
        cropped_image = origin_image.crop((0, 0, cropped_width, cropped_height))
        
        # load target image
        hr_image = Image.open(hr_image_path)
        hr_image.load()
        
        hr_image = hr_image.crop((0, 0, cropped_width * args.scale, cropped_height * args.scale))
        hr_tensor = TF.to_tensor(hr_image).unsqueeze(0)
        
        # get desired image size
        cropped_image.save(tmp_space, quality=args.quality)
        desired_size = os.path.getsize(tmp_space) - JPEG_HEADER_SIZE
        
        # encode as AdaSR
        q_importance = qn_module.infer_q(cropped_image)
        ada_image, ada_size = jpeg_adasr.enc(cropped_image, q_importance, tmp_space, desired_size)
        
        # find the JPEG (vanilla) QP that best matches our data size
        start_qp = max(5, args.quality-10)
        end_qp = min(100, args.quality+10)
        qp_range = range(start_qp, end_qp+1, 5)
        
        for i, qp in enumerate(qp_range):
            cropped_image.save(tmp_space, quality=qp)
            tmp_image = Image.open(tmp_space)
            tmp_size = os.path.getsize(tmp_space) - JPEG_HEADER_SIZE
            
            if i == 0:
                min_diff = abs(ada_size-tmp_size)
                base_size = tmp_size
                base_image = tmp_image
                base_image.load()
            else:
                diff = abs(ada_size-tmp_size)
                if min_diff > diff:
                    base_size = tmp_size
                    base_image = tmp_image
                    base_image.load()
                    min_diff = diff
        
        base_bpp = base_size * 8 / (base_image.width * base_image.height)
        ada_bpp = ada_size * 8 / (ada_image.width * ada_image.height)
        
        # SR inference (vanilla)
        base_sr_tensor_list = sr_inference(sr_models, base_image, args.split_by, device)
        
        base_psnr_list = []
        base_computing_list = []
        for ch_index, base_sr_tensor in enumerate(base_sr_tensor_list):
            base_psnr = calc_psnr(base_sr_tensor, hr_tensor)
            base_computing = model_to_mac[args.sr_model_name][sr_channel_list[ch_index]] * base_image.width * base_image.height
            base_psnr_list.append(base_psnr)
            base_computing_list.append(base_computing)
            
            if args.save_image:
                base_image = tensor2pil(base_sr_tensor)
                base_image.load()
                
                image_path = os.path.join(visual_result_dir, f'base_{sr_channel_list[ch_index]}_{base_name}')
                base_image.save(image_path)
        
        # SR inference (AdaSR)
        ada_sr_tensor_list = sr_inference(sr_models, ada_image, args.split_by, device)
        
        ada_psnr_list = []
        ada_computing_list = []
        for ch_index, ada_sr_tensor in enumerate(ada_sr_tensor_list):
            ada_psnr = calc_psnr(ada_sr_tensor, hr_tensor)
            ada_computing = model_to_mac[args.sr_model_name][sr_channel_list[ch_index]] * base_image.width * base_image.height
            ada_psnr_list.append(ada_psnr)
            ada_computing_list.append(ada_computing)
            
            if args.save_image:
                ada_image = tensor2pil(ada_sr_tensor)
                ada_image.load()
                
                image_path = os.path.join(visual_result_dir, f'ada_{sr_channel_list[ch_index]}_{base_name}')
                ada_image.save(image_path)
            
        avg_base_psnr += np.array(base_psnr_list)
        avg_base_computing += np.array(base_computing_list)
        avg_base_bpp += base_bpp
        
        avg_ada_psnr += np.array(ada_psnr_list)
        avg_ada_computing += np.array(ada_computing_list)
        avg_ada_bpp += ada_bpp
        
    avg_base_psnr = avg_base_psnr / len(lr_image_list)
    avg_base_computing = avg_base_computing / len(lr_image_list)
    avg_base_bpp = avg_base_bpp / len(lr_image_list)
    
    avg_ada_psnr = avg_ada_psnr / len(lr_image_list)
    avg_ada_computing = avg_ada_computing / len(lr_image_list)
    avg_ada_bpp = avg_ada_bpp / len(lr_image_list)
    
    df = pd.DataFrame(columns=['base_computing', 'base_psnr', 'base_bpp', 'ada_computing', 'ada_psnr', 'ada_bpp'])
    
    df['base_computing'] = avg_base_computing
    df['base_psnr'] = avg_base_psnr
    df['base_bpp'] = avg_base_bpp
    
    df['ada_computing'] = avg_ada_computing
    df['ada_psnr'] = avg_ada_psnr
    df['ada_bpp'] = avg_ada_bpp
        
    df.to_csv(log_path, sep='\t', index=False)
        
        
