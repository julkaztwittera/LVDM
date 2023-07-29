import os
import glob
import time
import argparse
import yaml, math
from tqdm import trange
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning import seed_everything
from decord import VideoReader, cpu

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from scripts.sample_utils import (load_model, 
                                  get_conditions, make_model_input_shape, torch_to_np, sample_batch, 
                                  save_results,
                                  save_args,
                                  )


# ------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--prompt", type=str, help="input text prompts for text2video (a sentence OR a txt file).")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    parser.add_argument("--vid_dir", type=str, help="path to the ground truth video/s", default="")
    # device args
    parser.add_argument("--ddp", action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--local_rank", type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--gpu_id", type=int, help="choose a specific gpu", default=0)
    # sampling args
    parser.add_argument("--n_samples", type=int, help="how many samples for each text prompt", default=2)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--decode_frame_bs", type=int, help="frame batch size for framewise decoding", default=1)
    parser.add_argument("--sample_type", type=str, help="ddpm or ddim", default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, help="ddim sampling -- number of ddim denoising timesteps", default=50)
    parser.add_argument("--eta", type=float, help="ddim sampling -- eta (0.0 yields deterministic sampling, 1.0 yields random sampling)", default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="fix a seed for randomness (If you want to reproduce the sample results)")
    parser.add_argument("--num_frames", type=int, default=16, help="number of input frames")
    parser.add_argument("--show_denoising_progress", action='store_true', default=False, help="whether show denoising progress during sampling one batch",)
    parser.add_argument("--cfg_scale", type=float, default=15.0, help="classifier-free guidance scale")
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    parser.add_argument("--return_recon", action='store_true', help="whether return reconstructions from autoencoder instead of samples", default=False)
    parser.add_argument("--return_gt", action='store_true', help="whether return ground truth video instead of samples", default=False)
    return parser

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_text2video(model, prompt, n_samples, batch_size,
                      sample_type="ddim", sampler=None, 
                      ddim_steps=50, eta=1.0, cfg_scale=7.5, 
                      decode_frame_bs=1,
                      ddp=False, all_gather=True, 
                      batch_progress=True, show_denoising_progress=False,
                      num_frames=None,
                      first_frame=None
                      ):
    # get cond vector
    assert(model.cond_stage_model is not None)
    cond_embd = get_conditions(prompt, model, batch_size)
    uncond_embd = get_conditions("", model, batch_size) if cfg_scale != 1.0 else None

    # sample batches
    all_videos = []
    n_iter = math.ceil(n_samples / batch_size)
    iterator  = trange(n_iter, desc="Sampling Batches (text-to-video)") if batch_progress else range(n_iter)
    for _ in iterator:
        noise_shape = make_model_input_shape(model, batch_size, T=num_frames)
        samples_latent = sample_batch(model, noise_shape, cond_embd,
                                            sample_type=sample_type,
                                            sampler=sampler,
                                            ddim_steps=ddim_steps,
                                            eta=eta,
                                            unconditional_guidance_scale=cfg_scale, 
                                            uc=uncond_embd,
                                            denoising_progress=show_denoising_progress,
                                            first_frame=first_frame
                                            )
        samples = model.decode_first_stage(samples_latent, decode_bs=decode_frame_bs, return_cpu=False)
        
        # gather samples from multiple gpus
        if ddp and all_gather:
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    
    all_videos = np.concatenate(all_videos, axis=0)
    assert(all_videos.shape[0] >= n_samples)
    return all_videos

def make_dataset(data_root):
    data_folder = data_root
    videos = glob.glob(os.path.join(data_folder, "**", f"*.mp4"), recursive=True)
    videos.sort()
    videos = videos[:20]
    print(f"NUMBER OF VIDEOS = {len(videos)}")
    return videos

@torch.no_grad()
def encode_first_frame(directory):
    video_path = directory
    caption_path = video_path.split(".")[0] + ".txt"
    try:
        video_reader = VideoReader(
            video_path,
            ctx=cpu(0),
            width=128,
            height=128,
        )
    except:
        print(f"Load video failed! path = {video_path}")

    resolution = [128, 128]
    video_length = 8
    frame_stride = 4
    all_frames = list(range(0, len(video_reader), frame_stride))
    if len(all_frames) < video_length:
        all_frames = list(range(0, len(video_reader), 1))

    # select random clip
    # rand_idx = random.randint(0, len(all_frames) - self.video_length)
    rand_idx = 0
    frame_indices = list(range(rand_idx, rand_idx + video_length, 1))
    frames = video_reader.get_batch(frame_indices)
    assert (
        frames.shape[0] == video_length
    ), f"{len(frames)}, self.video_length={video_length}"

    frames = torch.tensor(frames.asnumpy())
    first_frame = frames[0]
    frames = frames.permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
    assert (
        frames.shape[2] == resolution[0]
        and frames.shape[3] == resolution[1]
    ), f"frames={frames.shape}, self.resolution={resolution}"
    frames = (frames / 255 - 0.5) * 2

    with open(caption_path, "r") as file:
        caption = file.read()
    data = {"video": frames.unsqueeze(0), "caption": caption, "first_frame": first_frame}
    return data



# ------------------------------------------------------------------------------------------
def main():
    """
    text-to-video generation
    """
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    save_args(opt.save_dir, opt)
    
    # set device
    if opt.ddp:
        setup_dist(opt.local_rank)
        opt.n_samples = math.ceil(opt.n_samples / dist.get_world_size())
        gpu_id = None
    else:
        gpu_id = opt.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    
    # set random seed
    if opt.seed is not None:
        if opt.ddp:
            seed = opt.local_rank + opt.seed
        else:
            seed = opt.seed
        seed_everything(seed)

    # load & merge config
    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    print("config: \n", config)

    # get model & sampler
    model, _, _ = load_model(config, opt.ckpt_path)
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None
    
    # prepare prompt
    if opt.prompt.endswith(".txt"):
        opt.prompt_file = opt.prompt
        opt.prompt = None
    else:
        opt.prompt_file = None

    if opt.prompt_file is not None:
        f = open(opt.prompt_file, 'r')
        prompts, line_idx = [], []
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            if len(l) != 0:
                prompts.append(l)
                line_idx.append(idx)
        f.close()
        cmd = f"cp {opt.prompt_file} {opt.save_dir}"
        os.system(cmd)
    else:
        prompts = [opt.prompt]
        line_idx = [None]
    
    if len(opt.vid_dir) > 0:
        videos = make_dataset(opt.vid_dir)
        for video in videos:
            start = time.time()
            batch = encode_first_frame(video)
            prompt = batch["caption"]
            print(prompt)
            if len(prompt) > 200:
                continue

            if opt.return_gt:
                x_samples = batch["video"]
            else:
                N=1
                n_row=4
                with torch.no_grad():
                    z, c, x, xrec, xc, first_frame_encoded, img_c = model.get_input(
                        batch,
                        k=model.first_stage_key,
                        return_first_stage_outputs=True,
                        force_c_encode=True,
                        return_original_cond=True,
                        bs=N,
                        cond_key=None,
                    )

                if opt.return_recon:
                    x_samples = xrec
                else:
                    N = min(z.shape[0], N)
                    n_row = min(x.shape[0], n_row)

                    samples, _z_denoise_row = model.sample_log(
                        cond=c,
                        batch_size=N,
                        ddim=True,
                        ddim_steps=200,
                        eta=1.0,
                        temporal_length=16,
                        unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None,
                        first_frame_encoded=first_frame_encoded,
                        img_cond=img_c,
                    )
                    x_samples = model.decode_first_stage(samples, log_images=True)
            
            x_samples = torch_to_np(x_samples).numpy()

            if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
                prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
                save_name = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
                if opt.seed is not None:
                    save_name = save_name + f"_seed{seed:05d}"
                save_results(x_samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    else:
    # go
        start = time.time()  
        for prompt in prompts:
            # sample
            samples = sample_text2video(model, prompt, opt.n_samples, opt.batch_size,
                            sample_type=opt.sample_type, sampler=ddim_sampler,
                            ddim_steps=opt.ddim_steps, eta=opt.eta, 
                            cfg_scale=opt.cfg_scale,
                            decode_frame_bs=opt.decode_frame_bs,
                            ddp=opt.ddp, show_denoising_progress=opt.show_denoising_progress,
                            num_frames=opt.num_frames,
                            )
            # save
            if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
                prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
                save_name = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
                if opt.seed is not None:
                    save_name = save_name + f"_seed{seed:05d}"
                save_results(samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")

    if opt.ddp:
        dist.destroy_process_group()


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()