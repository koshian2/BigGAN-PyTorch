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

p = argparse.ArgumentParser(description="train CIFAR-10 basic")
p.add_argument("--base_ch", help="base chs of BigGAN-deep", default=128)
p.add_argument("--batch_size", help="batch size", default=128)
p.add_argument("--use_multi_gpu", help="flag of data parallel", default=True)

# base_ch = 128 (G=6,068,015, D=2,581,249)
# base_ch = 64 (G=2,149,295, D=651,649)
# base_ch = 32 (G=853,487, D=166,081)

args = p.parse_args()

def load_cifar(batch_size):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    utils.load_settings(args, "settings/cifar.json", cases)
    
    output_dir = f"cifar_case{cases}"

    device = "cuda" if args.use_multi_gpu else "cuda:1"
    torch.backends.cudnn.benchmark = True

    # update G = 40k
    nb_epoch = 200 * args.batch_size // 128 + 1  # 128:201, 256:401, ...
    sampling_period = args.batch_size // 128  # 128:1, 256:2, ...
    model_save_period = sampling_period * 5 # 128:5, 256:10, ...
    
    print("--- Conditions ---")
    print("- Case : ", cases)
    print(args)
    print("nb_epoch :", nb_epoch)
    
    dataloader = load_cifar(args.batch_size)

    model_G = Generator(args.base_ch, 32, 10, n_projected_dims=4)
    model_G_ema = Generator(args.base_ch, 32, 10, n_projected_dims=4) # cpu vertsion may fast
    model_D = Discriminator(args.base_ch, 32, 10)
    model_G, model_D = model_G.to(device), model_D.to(device)
    model_G_ema = model_G_ema.to(device)

    if args.use_multi_gpu:
        model_G, model_D = DataParallelWithCallback(model_G), DataParallelWithCallback(model_D)
        model_G_ema = DataParallelWithCallback(model_G_ema)

    param_G = torch.optim.Adam(model_G.parameters(), lr=5e-5, betas=(0, 0.999))
    param_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0, 0.999))

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(10).to(device)

    fake_img, fake_onehots = None, None
    ema = utils.EMA(model_G, model_G_ema)
    gan_loss = utils.HingeLoss(args.batch_size, device)
    update_G_counter = 1  # G:D=1:2
    
    def generate_fake_imgs(batch_len, labels):
        fake_onehots = labels.detach()
        x = truncnorm.rvs(-1.5, 1.5, size=(batch_len, 128)) # truncation trick = [-1.5, 1.5]
        rand_X = torch.FloatTensor(x).to(device)
        fake_img = model_G(rand_X, fake_onehots)
        return fake_img, fake_onehots

    for epoch in range(nb_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != args.batch_size: continue

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
        if epoch % sampling_period == 0:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            torchvision.utils.save_image(fake_img[:64], f"{output_dir}/epoch_{epoch:03}.png",
                                        nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))
                                        
        # save weights
        if epoch % model_save_period == 0:
            if not os.path.exists(output_dir + "/models"):
                os.mkdir(output_dir+"/models")
            utils.save_model(model_G, f"{output_dir}/models/gen_epoch_{epoch:03}.pytorch", args.use_multi_gpu)
            utils.save_model(model_G_ema, f"{output_dir}/models/gen_ema_epoch_{epoch:03}.pytorch", args.use_multi_gpu)
            utils.save_model(model_D, f"{output_dir}/models/dis_epoch_{epoch:03}.pytorch", args.use_multi_gpu)

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

def test(cases):
    utils.load_settings(args, "settings/cifar.json", cases)
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
    for i in range(6):
        test(i)
