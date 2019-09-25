import glob
from evaluation_metrics.inception_score import calculate_inception_score_given_tensor
from evaluation_metrics.fid_score import calculate_fid_given_tensor
from models.sync_batchnorm import DataParallelWithCallback
import pandas as pd
import torch
import os

def inceptions_score_fid_all(base_dir, generator_func, z_sampling_func, y_sampling_func, use_data_parallel,
                             n_minibatch_sampling, refrence_fid_statistics_path):
    model_paths = sorted(glob.glob(base_dir + "/models/gen_ema*.pytorch"))

    epochs = []
    inception_scores = []
    fids = []

    print(f"Calculating All Inception Scores / FIDs...  (# {len(model_paths)})")
    for i, path in enumerate(model_paths):
        model = generator_func()
        model.load_state_dict(torch.load(path))
        if use_data_parallel:
            model = DataParallelWithCallback(model)

        # generate images
        with torch.no_grad():
            imgs = []
            for _ in range(n_minibatch_sampling):
                z = z_sampling_func()
                y = y_sampling_func()
                x = model(z, y)
                imgs.append(x)
            imgs = torch.cat(imgs, dim=0).cpu()

        # eval_is
        iscore, _ = calculate_inception_score_given_tensor(imgs)
        # fid
        fid_score = calculate_fid_given_tensor(imgs, refrence_fid_statistics_path)
        # epoch
        epoch = int(os.path.basename(path).replace("gen_ema_epoch_", "").replace(".pytorch", ""))
        epochs.append(epoch)
        inception_scores.append(iscore)
        fids.append(fid_score)
        print(f"epoch = {epoch}, inception_score = {iscore}, fid = {fid_score}    [{i+1}/{len(model_paths)}]")

    df = pd.DataFrame({"epoch": epochs, "inception_score": inception_scores, "fid": fids})    
    df.to_csv(base_dir + "/inception_score.csv", index=False)
    