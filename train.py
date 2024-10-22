import torch
import os
import time
import random
from ChamferDistance import L2_ChamferDistance
from config import params
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ViPCDataLoader
import numpy as np
from models.PSSNet import PSSNet as Model
from models.utils import fps_subsample


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


opt = params()

if opt.cat != None:
    CLASS = opt.cat
else:
    CLASS = 'plane'

MODEL = 'PSS-Net'
FLAG = 'train'
BATCH_SIZE = int(opt.batch_size)
MAX_EPOCH = int(opt.n_epochs)
EVAL_EPOCH = int(opt.eval_epoch)
RESUME = opt.resume

TIME_FLAG = time.asctime(time.localtime(time.time()))
CKPT_RECORD_FOLDER = f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'
CONFIG_FILE = f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global_step = 0
best_loss = 99999
best_epoch = 0
resume_epoch = 0
model = Model().to(device)
loss_cd = L2_ChamferDistance()

ViPCDataset_train = ViPCDataLoader(
    './dataset/train_list.txt', data_path=opt.data_root, status="train", category=opt.cat)
train_loader = DataLoader(ViPCDataset_train,
                          batch_size=opt.batch_size,
                          num_workers=opt.nThreads,
                          shuffle=True,
                          drop_last=True)

ViPCDataset_test = ViPCDataLoader(
    './dataset/test_list.txt', data_path=opt.data_root, status="test", category=opt.cat)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=opt.batch_size,
                         num_workers=opt.nThreads,
                         shuffle=True,
                         drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=opt.lr)

if RESUME:
    ckpt_path = "./ckpt/avg.pt"
    ckpt_dict = torch.load(ckpt_path)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    resume_epoch = ckpt_dict['epoch']
    global_step = ckpt_dict['global_step']
    best_loss = ckpt_dict['loss']

length = len(train_loader)
lr_lambda = lambda epoch: 1 if global_step <= length*opt.lr_fix_epoch else 0.7 **((global_step-length*opt.lr_fix_epoch)/opt.steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)

if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

with open(CONFIG_FILE, 'w') as f:
    f.write('RESUME:'+str(RESUME)+'\n')
    f.write('FLAG:'+str(FLAG)+'\n')
    f.write('BATCH_SIZE:'+str(BATCH_SIZE)+'\n')
    f.write('MAX_EPOCH:'+str(MAX_EPOCH)+'\n')
    f.write('CLASS:'+str(CLASS)+'\n')
    f.write(str(opt.__dict__))

set_seed()
for epoch in range(resume_epoch, resume_epoch + opt.n_epochs+1):
    model.train()
    n_batches = len(train_loader)
    
    with tqdm(train_loader) as t:
        for batch_idx, data in enumerate(t):
            image = data[0].to(device)
            partial = data[2].to(device)
            gt = data[1].to(device)
            stage2 = fps_subsample(gt.contiguous(), 1024)
            stage1 = fps_subsample(stage2.contiguous(), 512)
            stage0 = fps_subsample(stage1.contiguous(), 256)
                          
            out = model(partial, image)
            
            loss_stage0 = loss_cd(stage0, out[0])
            loss_stage1 = loss_cd(stage1, out[1])
            loss_stage2 = loss_cd(stage2, out[2])
            loss_stage3 = loss_cd(gt, out[3])
            loss_stage0 = torch.mean(loss_stage0, 0)
            loss_stage1 = torch.mean(loss_stage1, 0)
            loss_stage2 = torch.mean(loss_stage2, 0)
            loss_stage3 = torch.mean(loss_stage3, 0)
            
            loss = loss_stage0 + loss_stage1 + loss_stage2 + loss_stage3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            
            scheduler.step()
            t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch, opt.n_epochs, batch_idx + 1, n_batches))
            t.set_postfix(loss = '%s' % ['%.4f' % l for l in [1e3 * loss_stage0.data.cpu(),
                                                              1e3 * loss_stage1.data.cpu(),
                                                              1e3 * loss_stage2.data.cpu(),
                                                              1e3 * loss_stage3.data.cpu()
                                                              ]])
    
    if epoch % EVAL_EPOCH == 0: 
        with torch.no_grad():
            model.eval()
            Loss = 0
            
            for data in tqdm(test_loader):
                image = data[0].to(device)
                partial = data[2].to(device)
                gt = data[1].to(device)
                
                out = model(partial, image)
                
                loss = loss_cd(out[-1], gt)
                
                Loss += loss * 1e3

            Loss = Loss/len(test_loader)

            if Loss < best_loss:
                best_loss = Loss
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': Loss
                }, f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}.pt')
                print('best epoch: ', best_epoch, 'cd: ', best_loss.item())
            print(epoch, ' ', Loss.item(), 'lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
            
print('Train Finished!!')
