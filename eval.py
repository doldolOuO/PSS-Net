import torch
from dataloader import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.PSSNet import PSSNet as Model
from ChamferDistance import L2_ChamferDistance, F1Score


category = "cellphone"
# plane cabinet car chair lamp couch table watercraft
# bench monitor speaker cellphone

ckpt_dir = "ckpt/avg.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_root = "/data/FCH/data/ShapeNetViPC_2048"

ViPCDataset_test = ViPCDataLoader('./dataset/test_list.txt', data_path=data_root, status="test", category = category)
test_loader = DataLoader(ViPCDataset_test,
                            batch_size=50,
                            num_workers=8,
                            shuffle=False,
                            drop_last=False)

model = Model().to(device)
model.load_state_dict(torch.load(ckpt_dir)['model_state_dict'])

loss_eval = L2_ChamferDistance()
loss_f1 = F1Score()

with torch.no_grad():
    model.eval()
    i = 0
    Loss = 0 
    f1_final = 0
    for data in tqdm(test_loader):
        i += 1
        image = data[0].to(device)
        partial = data[2].to(device)
        gt = data[1].to(device)
        out = model(partial, image)

        #Compute the eval loss
        loss = loss_eval(out[-1], gt)
        f1, _, _  = loss_f1(out[-1], gt)
        f1 = f1.mean()
        Loss += loss * 1e3
        f1_final += f1
        
    Loss = Loss/i
    f1_final = f1_final/i

print(f"The evaluation loss for {category} is :{Loss}")
print(f"The F1-score for {category} is :{f1_final}")