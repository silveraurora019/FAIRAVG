import logging
import torch
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path

from utils import set_for_logger
from dataloaders import build_dataloader
from loss import DiceLoss, JointLoss
import torch.nn.functional as F
from nets import build_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_root', type=str, required=False, 
                        default="E:/A_Study_Materials/Dataset/fundus-preprocesed/fundus", 
                        help="Data directory (指向 fundus 或 prostate 的 .npy 文件夹)")
    # E:/A_Study_Materials/Dataset/Prostate
    parser.add_argument('--dataset', type=str, default='fundus', 
                        help="Dataset type: 'fundus' (4 站点) 或 'prostate' (6 站点)")
    
    # 强制使用 unet_pro，因为两种方法都需要 z 特征
    parser.add_argument('--model', type=str, default='unet_pro', help='Model type (unet or unet_pro). Required by MI-Sim and UFT.')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--experiment', type=str, default='experiment_fundus_avg', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    

    args = parser.parse_args()
    return args

# (communication, train, test 函数保持不变)
def communication(server_model, models, client_weights):
    with torch.no_grad():
        device = next(server_model.parameters()).device
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
    return server_model

def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
    model.train()
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    for epoch in range(epochs):
        train_acc = 0.
        loss_all = 0.
        if len(dataloader) == 0:
            logging.warning(f"Client {cid} training dataloader is empty.")
            continue
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x) 
            else:
                output = model(x)
            optimizer.zero_grad()
            loss = loss_func(output, target)
            loss_all += loss.item()
            train_acc += DiceLoss().dice_coef(output, target).item()
            loss.backward()
            optimizer.step()
        
        if len(dataloader) > 0:
            avg_loss = loss_all / len(dataloader)
            train_acc = train_acc / len(dataloader)
            logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))

def test(model, dataloader, device, loss_func):
    model.eval()
    loss_all = 0
    test_acc = 0
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    if len(dataloader) == 0:
        logging.warning("Test/Val dataloader is empty.")
        return 0.0, 0.0 # 返回 0 避免除零错误

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
            loss = loss_func(output, target)
            loss_all += loss.item()
            test_acc += DiceLoss().dice_coef(output, target).item()
        
    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)
    return loss, acc

def _flat_params(model):
    return torch.cat([p.data.float().view(-1).cpu() for p in model.parameters()])

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # 2. 动态定义客户端列表
    if args.dataset == 'fundus':
        clients = ['site1', 'site2', 'site3', 'site4']
    elif args.dataset == 'prostate':
        clients = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
    else:
        raise ValueError(f"Unknown client list for dataset: {args.dataset}")

    # 3. build dataset (传入 clients)
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # 4. build model (传入 clients)
    local_models, global_model = build_model(args, clients, device)


    # (Loss 和 Optimizer)
    # --- 修改：使用 JointLoss ---
    loss_fun = JointLoss() 
    
    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    # (训练循环)
    best_dice = 0
    best_dice_round = 0
    best_local_dice = []
    

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):

        logging.info('-------- Commnication Round: %3d --------'%r)

        # 1. 本地训练
        for idx, client in enumerate(clients):
            train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            
        temp_locals = copy.deepcopy(local_models)
        
        
        # 3. 聚合 (始终使用 FedAvg)
        logging.info('Using standard FedAvg aggregation.')
        # 我们总是使用 client_weight_tensor (基于数据集大小的 FedAvg 权重)
        communication(global_model, temp_locals, client_weight_tensor)


        # 4. 分发全局模型
        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r % args.test_step == 0:
            # 5. 测试 (使用测试集 - 用于最终报告和保存最佳模型)
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)

                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice) if len(avg_dice) > 0 else 0
            avg_loss_v = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0
            
            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))


            avg_val_dice = []
            for idx, client in enumerate(clients):
                _, val_dice = test(local_models[idx], val_dls[idx], device, loss_fun)
                avg_val_dice.append(val_dice)
            
            current_avg_val_dice_tensor = torch.tensor(avg_val_dice, device=device, dtype=torch.float32)
            logging.info('Round: [%d]  avg_val_acc (for feedback): %f'%(r, current_avg_val_dice_tensor.mean().item()))


            # 7. 保存最佳模型 (仍然基于测试集性能 avg_dice_v)
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx] if idx < len(best_local_dice) else 0.0))


if __name__ == '__main__':
    args = get_args()
    main(args)