"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import os
import numpy as np
import torch
from terminaltables import AsciiTable


def load_checkpoint(cfg, model):
    ckpt_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'snapshot')
    if cfg.TEST.test_epoch is None:  # the lastest epoch
        all_ckpts = list(filter(lambda x: x.endswith('.pth'), os.listdir(ckpt_dir)))
        all_epochs = [int(filename.split('.')[-2].split('_')[-1]) for filename in all_ckpts]
        fids = np.argsort(all_epochs)
        ckpt_file = os.path.join(ckpt_dir, all_ckpts[fids[-1]])
        test_epoch = all_epochs[fids[-1]]
    else:
        assert isinstance(cfg.TEST.test_epoch, int)
        ckpt_file = os.path.join(ckpt_dir, cfg.TRAIN.snapshot_prefix + '%02d.pth'%(cfg.TEST.test_epoch))
        test_epoch = cfg.TEST.test_epoch
    print('Loading the model checkpoint: {}'.format(ckpt_file))
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model'])
    return model, test_epoch


def print_de_table(all_ades, all_fdes, subset):
    mADE = np.mean(list(all_ades.values()))
    mFDE = np.mean(list(all_fdes.values()))
    print("mADE({}) = {:.3f}, mFDE({}) = {:.3f}".format(subset, mADE, subset, mFDE))
    # print as table
    display = [list(all_ades.keys()), ["{:.3f} / {:.3f}".format(v1, v2) for v1, v2 in zip(all_ades.values(), all_fdes.values())]]
    table = AsciiTable(display)
    table.inner_footing_row_border = True
    table.justify_columns = {i: 'center' for i in range(len(all_ades))}
    print(table.table)


def print_delta_table(all_dxs, all_dys, all_dzs, subset):
    mDX = np.mean(list(all_dxs.values()))
    mDY = np.mean(list(all_dys.values()))
    print_str = "mDX({}) = {:.3f}, mDY({}) = {:.3f}".format(subset, mDX, subset, mDY)
    if len(all_dzs) > 0:
        mDZ = np.mean(list(all_dzs.values()))
        print_str += ", mDZ({}) = {:.3f}".format(subset, mDZ)
    print(print_str)
    # print as table
    if len(all_dzs) > 0:
        display = [list(all_dxs.keys()), ["{:.3f} / {:.3f} / {:.3f}".format(v1, v2, v3) 
                                      for v1, v2, v3 in zip(all_dxs.values(), all_dys.values(), all_dzs.values())]]
    else:  # 2D targets
        display = [list(all_dxs.keys()), ["{:.3f} / {:.3f}".format(v1, v2) 
                                      for v1, v2 in zip(all_dxs.values(), all_dys.values())]]
    table = AsciiTable(display)
    table.inner_footing_row_border = True
    table.justify_columns = {i: 'center' for i in range(len(all_dxs))}
    print(table.table)