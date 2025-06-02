import sys
import os
import argparse
import time
import yaml
import os
sys.path.append('.')

if __name__ == '__main__':
    with open('options/traineval_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # close ddp temporarily
    cuda_devices = config['eval'].get('CUDA_VISIBLE_DEVICES', '0')
    resume_path = config['eval'].get('resume', ' ')    

    if os.path.exists(resume_path):
        COMMANDLINE = f"CUDA_VISIBLE_DEVICES={cuda_devices} python traineval.py --evaluate --resume={resume_path}"
    else:
        print("found invalid model path:\n", resume_path)
        COMMANDLINE = f"CUDA_VISIBLE_DEVICES={cuda_devices} python traineval.py --evaluate"
    
    print(COMMANDLINE)
    os.system(COMMANDLINE)