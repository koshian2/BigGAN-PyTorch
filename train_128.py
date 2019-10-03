import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import numpy as np
from scipy.stats import truncnorm
import argparse
import glob

import utils
from models.biggan_deep import Generator, Discriminator
from models.sync_batchnorm import DataParallelWithCallback
from evaluate import inceptions_score_fid_all

p = argparse.ArgumentParser(description="train 128x128")
p.add_argument("--dataset", help="name of dataset", default="anime")
p.add_argument("--data_root_dir", help="root directory of data", default="thumb")
p.add_argument("--n_epoch", help="# epochs", default=1)
p.add_argument("--n_classes", help="# classes", default=176)
p.add_argument("--n_projected_dims", help="# projected onehot dims", default=32)

args = p.parse_args()

def load_dataset(batch_size):
    trans = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root="./data/"+args.data_root_dir, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    utils.load_settings(args, "settings/res128.json", cases)
    
    output_dir = f"res128_case{cases}"

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    
    print("--- Conditions ---")
    print("- Case : ", cases)
    print(args)
    
    batch_size = 64
    dataloader = load_dataset(batch_size)

    model_G = Generator(64, 128, args.n_classes, n_projected_dims=args.n_projected_dims)
    model_G_ema = Generator(64, 128, args.n_classes, n_projected_dims=args.n_projected_dims)
    model_D = Discriminator(64, 128, args.n_classes)
    model_G, model_D = model_G.to(device), model_D.to(device)
    model_G_ema = model_G_ema.to(device)

    model_G, model_D = DataParallelWithCallback(model_G), DataParallelWithCallback(model_D)
    model_G_ema = DataParallelWithCallback(model_G_ema)

    param_G = torch.optim.Adam(model_G.parameters(), lr=5e-5, betas=(0, 0.999))
    param_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0, 0.999))

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(args.n_classes).to(device)

    fake_img, fake_onehots = None, None
    ema = utils.EMA(model_G, model_G_ema)
    gan_loss = utils.HingeLoss(batch_size, device)
    update_G_counter = 1  # G:D=1:2
    
    def generate_fake_imgs(batch_len, labels):
        fake_onehots = labels.detach()
        x = truncnorm.rvs(-1.5, 1.5, size=(batch_len, 128)) # truncation trick = [-1.5, 1.5]
        rand_X = torch.FloatTensor(x).to(device)
        fake_img = model_G(rand_X, fake_onehots)
        return fake_img, fake_onehots

    for epoch in range(args.n_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != batch_size: continue

            real_img = real_img.to(device)
            real_onehots = onehot_encoding[labels.to(device)]
                        
            # train D
            param_G.zero_grad()
            param_D.zero_grad()
            # train real
            d_out_real = model_D(real_img, real_onehots)
            loss_real = gan_loss(d_out_real, "dis_real")
            # train fake
            if fake_img is None:
                fake_img, fake_onehots = generate_fake_imgs(batch_len, real_onehots)
                fake_img = fake_img.detach()
            d_out_fake = model_D(fake_img, fake_onehots)
            loss_fake = gan_loss(d_out_fake, "dis_fake")
            # fake + real loss
            loss = loss_real + loss_fake
            log_loss_D.append(loss.item())

            # backprop
            loss.backward()
            param_D.step()

            # train G
            if update_G_counter == 0:
                param_G.zero_grad()
                param_D.zero_grad()

                fake_img_g, fake_onehots = generate_fake_imgs(batch_len, real_onehots)
                fake_img = fake_img_g.detach()
                g_out = model_D(fake_img_g, fake_onehots)

                loss = gan_loss(g_out, "gen")
                log_loss_G.append(loss.item())  # loss without orthogonal regularization

                # backprop
                loss.backward()
                # orthogonal regularization
                utils.orthogonal_regularization(model_G)
                # update G
                param_G.step()
                # update EMA
                ema.update()
                # reset counter
                update_G_counter = 1
            else:
                update_G_counter -= 1

        # log
        result["d_loss"].append(statistics.mean(log_loss_D))
        result["g_loss"].append(statistics.mean(log_loss_G))
        print(f"epoch = {epoch}, g_loss = {result['g_loss'][-1]}, d_loss = {result['d_loss'][-1]}")        
            
        # save screen shot
        if epoch % 1 == 0:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            torchvision.utils.save_image(fake_img[:25], f"{output_dir}/epoch_{epoch:03}.png",
                                        nrow=5, padding=3, normalize=True, range=(-1.0, 1.0))
                                        
        # save weights
        if epoch % 5 == 0:
            if not os.path.exists(output_dir + "/models"):
                os.mkdir(output_dir+"/models")
            utils.save_model(model_G, f"{output_dir}/models/gen_epoch_{epoch:03}.pytorch", True)
            utils.save_model(model_G_ema, f"{output_dir}/models/gen_ema_epoch_{epoch:03}.pytorch", True)
            utils.save_model(model_D, f"{output_dir}/models/dis_epoch_{epoch:03}.pytorch", True)

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

def test(cases):
    utils.load_settings(args, "settings/res128.json", cases)
    def g_func():
        x = Generator(args.base_ch, 32, 10, n_projected_dims=4).cuda()
        return x
    def z_func():
        return torch.FloatTensor(truncnorm.rvs(-1.5, 1.5, size=(args.batch_size, 128))).cuda()
    def y_func():
        return torch.eye(10)[torch.randint(0, 10, (args.batch_size,))].cuda()    
    inceptions_score_fid_all("cifar_case" + str(cases), g_func, z_func, y_func, args.use_multi_gpu,
                             50000//args.batch_size+1, "cifar10_train.pkl")

if __name__ == "__main__":
    train(0)
