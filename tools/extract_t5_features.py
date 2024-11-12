import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import torch
import numpy as np
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from pathlib import Path

from diffusion.model.t5 import T5Embedder

def extract_caption_t5_do(q):
    while not q.empty():
        item = q.get()
        extract_caption_t5_job(item)
        q.task_done()

def extract_caption_t5_job(item):
    global mutex
    global t5
    # global t5_save_dir

    with torch.no_grad():
        with open(item, 'r') as file:
            caption = file.readline().strip()
        # caption = item['prompt'].strip()
        if isinstance(caption, str):
            caption = [caption]

        # save_path = item.replace('.caption','')
        # save_path = os.path.join(args.save_npz_folder, os.path.basename(item).replace('.caption','.npz'))
        save_path = os.path.join(args.save_npz_folder, os.path.basename(item).replace('.txt','.npz'))
        if os.path.exists(f"{save_path}.npz"):
            return
        try:
            mutex.acquire()
            caption_emb, emb_mask = t5.get_text_embeddings(caption)
            mutex.release()
            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
        except Exception as e:
            print(e)

def get_caption_files(folder_path):
    caption_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                caption_files.append(os.path.join(root, file))
    caption_files.sort()
    return caption_files

def extract_caption_t5():
    global t5
    t5 = T5Embedder(device="cuda",dir_or_name='', local_cache=True, cache_dir=args.t5_ckpt, model_max_length=120)

    caption_files = get_caption_files(args.caption_folder)
    
    global mutex
    mutex = threading.Lock()
    jobs = Queue()

    for item in tqdm(caption_files):
        jobs.put(item)

    for _ in range(20):
        worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
        worker.start()

    jobs.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--t5_ckpt', type=str, default=None)
    parser.add_argument('--caption_folder', type=str, default=None)
    parser.add_argument('--save_npz_folder', type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_npz_folder, exist_ok=True)

    extract_caption_t5()