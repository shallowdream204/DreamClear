'''
 * DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation
 * Modified from PixArt-alpha by Yuang Ai
 * 13/10/2024
'''
import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
import torch.nn.functional as F

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType

from diffusion import IDDPM, DPMS
from diffusion.model.builder import build_model
from diffusion.model.nets import PixArtMS, ControlPixArtMSHalfSR2Branch, SwinIR
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed
from diffusers.models import AutoencoderKL

from PIL import Image
from torchvision.utils import save_image
from diffusion.utils.align_color import wavelet_reconstruction, adaptive_instance_normalization


warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'

@torch.no_grad()
def log_validation(model, accelerator, device):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    save_dir = os.path.join(config.work_dir,'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,'input'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'output'), exist_ok=True)

    hw = torch.tensor([[1024, 1024]], dtype=weight_dtype, device=device).repeat(1, 1)
    ar = torch.tensor([[1.]],dtype=weight_dtype, device=device).repeat(1, 1)
    path_lq = args.image_path
    files = os.listdir(path_lq)
    files.sort()
    for i in range(len(files)):
        file = os.path.join(path_lq, files[i])
        txt_info = np.load(os.path.join(args.npz_path,files[i].replace('.png','.npz')))
        txt_fea = torch.from_numpy(txt_info['caption_feature']).to(device) #1 120 4096
        txt_fea = txt_fea[None] 
        attention_mask = torch.ones(1, 1, txt_fea.shape[1]) 
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None].to(device) # 1 1 120
        attention_mask = attention_mask[None]
        img_lq = Image.open(file).convert('RGB')
        img_lq.save(os.path.join(save_dir, 'input', files[i]))

        img_lq = np.array(img_lq)
        img_lq = torch.tensor(img_lq, dtype=vae_dtype, device=device) / 255.0
        img_lq = img_lq.permute(2, 0, 1).unsqueeze(0).contiguous()
        img_lq  = F.interpolate(
                        img_lq,
                        size=(args.sr_size,args.sr_size),
                        mode='bicubic',
                        )
        img_pre = swinir(img_lq)

        img_pre = torch.clamp(img_pre, 0, 1.0)

        posterior_lq = vae.encode(2*img_lq-1.0)
        latent_lq = posterior_lq.latent_dist.mode()*config.scale_factor
        latent_lq.to(dtype=weight_dtype)
        posterior_pre = vae.encode(2*img_pre-1.0)
        latent_pre = posterior_pre.latent_dist.mode()*config.scale_factor
        latent_pre.to(dtype=weight_dtype)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=attention_mask,c_lq=latent_lq,c_pre=latent_pre)

        if args.sampling_alg == 'dpm':
            z = torch.randn(1, 4, args.sr_size//8, args.sr_size//8, device=device,dtype=weight_dtype)
            null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(device)
            dpm_solver = DPMS(  model.forward_with_dpmsolver,
                                condition=txt_fea,
                                uncondition=null_y,
                                cfg_scale=args.cfg_scale,
                                model_kwargs=model_kwargs)
            with torch.cuda.amp.autocast():
                samples = dpm_solver.sample(
                    z,
                    steps=50,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                    )
        elif args.sampling_alg == 'iddpm':
            if args.lre:
                t_diffusion = IDDPM(str(1000))
                noise=torch.randn_like(latent_pre, device=device,dtype=weight_dtype)
                timesteps = torch.randint(args.start_point, args.start_point+1, (1,), device=latent_pre.device).long()
                z = t_diffusion.q_sample(latent_pre,timesteps,noise=noise).repeat(2, 1, 1, 1)
            else:
                z = torch.randn(1, 4, args.sr_size//8, args.sr_size//8, device=device,dtype=weight_dtype).repeat(2, 1, 1, 1)
            null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(device)

            model_kwargs = dict(y=torch.cat([txt_fea, null_y]), cfg_scale=args.cfg_scale,
                                data_info={'img_hw': hw, 'aspect_ratio': ar},
                                mask=attention_mask, c_lq=latent_lq,c_pre=latent_pre)

            diffusion = IDDPM(str(50))
            with torch.cuda.amp.autocast():
                samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device, accelerator=accelerator
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
        samples = vae.decode(samples.to(dtype=vae_dtype) / config.scale_factor).sample
        torch.cuda.empty_cache()
        hq = (samples+1)/2
        if args.color_align == 'wavelet':
            hq = wavelet_reconstruction(hq,img_pre)
        elif args.color_align == 'adain':
            hq = adaptive_instance_normalization(hq,img_pre)
        hq = hq.clamp(0, 1.0)
        save_image(hq[0], os.path.join(save_dir,'output', files[i]), nrow=1, normalize=True, value_range=(0, 1))

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true')
    parser.add_argument('--resume_lr_scheduler', action='store_true')

    # Related to Inference
    parser.add_argument('--dreamclear_ckpt', type=str, default=None)
    parser.add_argument('--swinir_ckpt', type=str, default=None)
    parser.add_argument('--vae_ckpt', type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default='bf16')
    parser.add_argument('--lre', action='store_true')
    parser.add_argument('--start_point', type=int, default=999)
    parser.add_argument('--sr_size', type=int, default=1024)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument("--color_align", type=str, choices=['wavelet', 'adain'], default='wavelet')
    parser.add_argument("--sampling_alg", type=str, choices=['iddpm', 'dpm'], default='iddpm')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--npz_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.save_dir is not None:
        config.work_dir = args.save_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.data_root:
        config.data_root = args.data_root
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=args.resume_optimizer,
            resume_lr_scheduler=args.resume_lr_scheduler)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 6
        config.optimizer.update({'lr': args.lr})

    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=9600)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, 'logs'),
        fsdp_plugin=fsdp_plugin,
        # even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )
    logger = get_root_logger(os.path.join(config.work_dir, 'eval_dreamclear.log'))

    if args.seed is not None:
        args.seed = init_random_seed(args.seed)
        set_random_seed(args.seed)

    logger.info(f"Initializing: {init_train} for inference")
    image_size = config.image_size  # @param [512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps))
    model: PixArtMS = build_model(config.model,
                                  config.grad_checkpointing,
                                  config.get('fp32_attention', False),
                                  input_size=latent_size,
                                  learn_sigma=learn_sigma,
                                  pred_sigma=pred_sigma,
                                  **model_kwargs)

    model: ControlPixArtMSHalfSR2Branch = ControlPixArtMSHalfSR2Branch(model, copy_blocks_num=config.copy_blocks_num).eval()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info("Using fp16 inference for DiT.")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info("Using bf16 inference for DiT.")
    vae_dtype = torch.float32

    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"T5 max token length: {config.model_max_length}")

    vae = AutoencoderKL.from_pretrained(args.vae_ckpt)
    for param in vae.parameters():
        param.requires_grad_(False)
    vae.eval()
    vae = vae.to(accelerator.device, dtype=vae_dtype)

    swinir = SwinIR( img_size= 64,patch_size= 1,in_chans= 3,embed_dim= 180,depths= [6, 6, 6, 6, 6, 6, 6, 6],num_heads= [6, 6, 6, 6, 6, 6, 6, 6],
        window_size= 8,mlp_ratio= 2,sf= 8,img_range= 1.0,upsampler= "nearest+conv",resi_connection= "1conv",unshuffle= True,unshuffle_scale= 8)
    ckpt = torch.load(args.swinir_ckpt,map_location="cpu")['state_dict']
    new_ckpt = {}
    for key, value in ckpt.items():
        new_key = key.replace('module.', '') 
        new_ckpt[new_key] = value
    del ckpt
    swinir.load_state_dict(new_ckpt)
    del new_ckpt
    for param in swinir.parameters():
        param.requires_grad_(False)
    swinir.eval()
    swinir = swinir.to(accelerator.device)

    if config.full_mixed:
        model.to(device=accelerator.device,dtype=weight_dtype)
        if accelerator.mixed_precision == "fp16":
            def patch_accelerator_for_fp16_training(accelerator):
                org_unscale_grads = accelerator.scaler._unscale_grads_

                def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                    return org_unscale_grads(optimizer, inv_scale, found_inf, True)

                accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
            patch_accelerator_for_fp16_training(accelerator)

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)


    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    if args.dreamclear_ckpt is not None:
        missing, unexpected = load_checkpoint(args.dreamclear_ckpt, model)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    model = accelerator.prepare(model,)
    log_validation(model,accelerator,model.device)