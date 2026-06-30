import os
import time
import torch
import itertools
import argparse
import glob
import os
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import os
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import librosa

import multiprocessing, functools, json, glob
import subprocess, time
from subprocess import TimeoutExpired
import argparse, sys, glob
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
import json, os



import string
import argparse
from collections import defaultdict
import sys
import time, random
import json, os, glob, random
import multiprocessing
from multiprocessing import set_start_method
import os
args = None
program_name = sys.argv[0]


def get_program_name():
  global program_name
  return program_name

def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Parse rank and world size.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process (default: 0)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (default: 1)")
    parser.add_argument("--run",  action="store_true",  help="(Re)start the processing")    
    args = parser.parse_args()
    return args

def get_rank() -> int:
    try:
        rank = int(os.environ['SLURM_PROCID'])
    except:
        rank = args.rank
    return rank


def _get_tasks_per_node() -> int:
    try:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    except:
        return 1


def _get_num_nodes() -> int:
  try:
    return int(os.environ['SLURM_JOB_NUM_NODES'])
  except:
    return args.world_size


def get_world_size() -> int:
    return _get_num_nodes() * _get_tasks_per_node()

import numpy as np
from snac import SNAC
num_devices = torch.cuda.device_count()
device  = "cuda:0"
if num_devices == 0:
    num_devices = 1
    device = "cpu"

model = None
fem_edu = None

def process(arg):
  
    global model, device, fem_edu
    file, i = arg
    if model is None:
      device = device.split(":")[0]
      if device == "cuda":
        device = "cuda:"+str(i)
      model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir="/leonardo_work/EUHPC_E03_068/.cache").eval().to(device)
    if not fem_edu:
      fem_edu = json.load(open("fem_edu.jsonl"))
    file2 = file.split("/")[-1]
    if os.path.exists(file.replace(".jsonl", ".reduced")): return
    if True: # with open(file.replace(".jsonl", ".tmp1"), "w") as outf:
        for l in tqdm(open(file)):
            data = json.loads(l)
            if type(data['media']) is str:
              data['media'] = json.loads(data['media'])
            for idx, path in  list(data['media'].items()):
              if "audio" not in idx: continue
              if os.path.exists(root_dir+"/"+path.replace(".ogg", ".listen")): continue
              id = data['metadata']['params']
              if type(id) is str:
                id = json.loads(id)
              id = id['id']
              #if not os.path.exists(root_dir+"/"+path+"_clone") and id not in fem_edu:
              #  continue
              #print (data)
              TARGET_SAMPLE_RATE = 24000
              audio, _ = librosa.load(root_dir+"/"+path, sr=TARGET_SAMPLE_RATE, mono=True)
              if audio.dtype != np.float32:
                  audio = audio.astype(np.float32)
              audio = torch.Tensor(audio).unsqueeze(0).unsqueeze(0).to(device)
              with torch.inference_mode():
                  codes = model.encode(audio)
                  #audio_hat = model.decode(codes)
                  num_base_tokens = codes[0].shape[1]
                  b = 0

                  speak = []
                  listen = []          
                  for i in range(num_base_tokens):
                      idx1_a=2*i; idx1_b=(2*i)+1; idx2_a=4*i; idx2_b=(4*i)+1; idx2_c=(4*i)+2; idx2_d=(4*i)+3
                      if (idx1_b < codes[1].shape[1] and idx2_d < codes[2].shape[1]):
                          current_group = [
                              codes[0][b, i].item() + 128266,
                              codes[1][b, idx1_a].item() + 128266 + 4096,
                              codes[2][b, idx2_a].item() + 128266 + (2*4096),
                              codes[2][b, idx2_b].item() + 128266 + (3*4096),
                              codes[1][b, idx1_b].item() + 128266 + (4*4096),
                              codes[2][b, idx2_c].item() + 128266 + (5*4096),
                              codes[2][b, idx2_d].item() + 128266 + (6*4096)
                          ]
                          speak.extend(current_group)
                          current_group = [
                              codes[0][b, i].item() + 128266,
                              codes[1][b, idx1_a].item() + 128266 + 4096,
                              codes[1][b, idx1_b].item() + 128266 + (4*4096),
                          ]

                          listen.extend(current_group)
                  print (root_dir+"/"+path.replace(".ogg", ".listen"))
                  json.dump(listen, open(root_dir+"/"+path.replace(".ogg", ".listen"), "w"))
                  json.dump(speak, open(root_dir+"/"+path.replace(".ogg", ".speak"), "w"))
                  if not os.path.exists(root_dir+"/"+path+"_clone"):
                    continue
                  audio, _ = librosa.load(root_dir+"/"+path+"_clone", sr=TARGET_SAMPLE_RATE, mono=True)
                  if audio.dtype != np.float32:
                      audio = audio.astype(np.float32)
                  audio = torch.Tensor(audio).unsqueeze(0).unsqueeze(0).to(device)
                  with torch.inference_mode():
                      codes = model.encode(audio)
                      #audio_hat = model.decode(codes)
                      num_base_tokens = codes[0].shape[1]
                      b = 0

                      speak = []
                      listen = []          
                      for i in range(num_base_tokens):
                          idx1_a=2*i; idx1_b=(2*i)+1; idx2_a=4*i; idx2_b=(4*i)+1; idx2_c=(4*i)+2; idx2_d=(4*i)+3
                          if (idx1_b < codes[1].shape[1] and idx2_d < codes[2].shape[1]):
                              current_group = [
                                  codes[0][b, i].item() + 128266,
                                  codes[1][b, idx1_a].item() + 128266 + 4096,
                                  codes[2][b, idx2_a].item() + 128266 + (2*4096),
                                  codes[2][b, idx2_b].item() + 128266 + (3*4096),
                                  codes[1][b, idx1_b].item() + 128266 + (4*4096),
                                  codes[2][b, idx2_c].item() + 128266 + (5*4096),
                                  codes[2][b, idx2_d].item() + 128266 + (6*4096)
                              ]
                              speak.extend(current_group)
                              current_group = [
                                  codes[0][b, i].item() + 128266,
                                  codes[1][b, idx1_a].item() + 128266 + 4096,
                                  codes[1][b, idx1_b].item() + 128266 + (4*4096),
                              ]

                              listen.extend(current_group)
                      print (root_dir+"/"+path.replace(".ogg", ".clone_listen"))
                      json.dump(listen, open(root_dir+"/"+path.replace(".ogg", ".clone_listen"), "w"))
                      json.dump(speak, open(root_dir+"/"+path.replace(".ogg", ".clone_speak"), "w"))                  
                    
    with open(file.replace(".jsonl", ".reduced"), "w") as outf: pass




root_dir = "/leonardo_work/EUHPC_E03_068/datasets/working/valid/data/"    

def subprocess_and_monitor(new_args):
    """
    Starts the subprocess and monitors its output.
    """
    global args
    args = new_args
    world_size = get_world_size()
    rank = get_rank()
    total_shards = len(list([f for f in glob.glob(f"{root_dir}/*.jsonl")]))
    process = subprocess.Popen(['python3', get_program_name(), '--rank', str(rank), '--world_size', str(world_size), '--run'])    
    while True:
        try:
            num = len(list(glob.glob(f"{root_dir}/*.reduced")))          

            time.sleep(30)
            outs, errs = process.communicate(timeout=30)
            if outs:
                outs = outs.decode().strip()
                if "error" in outs:
                    logger.warning(outs)
            processed_shards = len(list(glob.glob(f"{root_dir}/*.reduced")))
            if process.returncode != 0 or not outs or  num ==  processed_shards:
                # uploading stalled, or there is no output or the uploading program returned an error code. 
                process.kill()
                outs, errs = process.communicate()
                time.sleep(10)                
                process = subprocess.Popen(['python3', get_program_name(), '--rank', str(rank), '--world_size', str(world_size), '--run'])    
        except TimeoutExpired:
                process.kill()
                outs, errs = process.communicate()
                time.sleep(10)                                
                process = subprocess.Popen(['python3', get_program_name(), '--rank', str(rank), '--world_size', str(world_size), '--run'])    
def run():
    global all_files
    # Iterate through all files in the directory and subdirectories
    all_files = list(glob.glob(root_dir + '*.jsonl'))
    ws = get_world_size()
    rank = get_rank()
    print ("starting rank", rank)
    rank2files = {}
    j = -1
    for file in all_files:
        j += 1
        for k in range(ws):
            if j == k:
                p = rank2files[k] = rank2files.get(k,[])
                p.append(file)
                if j == ws-1:
                    j = -1
                break
    files = rank2files[rank]
    files = [file for file in files if not os.path.exists(file.replace(".jsonl", ".reduced"))]
    random.shuffle(files)
    files = [(file, i%4) for i, file in enumerate(files)]
    with multiprocessing.Pool(4) as p:
      for _ in p.imap_unordered(process, files):
        pass

if __name__ == "__main__":
    args = parse_args()
    run()


                        
