import torch

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

def tensor_to_dataset(image_tensor, verbose=False):
    assert image_tensor.size(1) == 3
    # Range check
    with torch.no_grad():
        minvalue = torch.min(image_tensor)
        maxvalue = torch.max(image_tensor)
        if minvalue <= -1.01 or minvalue >= -0.98:
            print(f"Image tensor should be [-1, 1] range. Min value = {minvalue}. Do you intended ?")
        if maxvalue >= 1.01 or maxvalue <= 0.98:
            print(f"Image tensor should be [-1, 1] range. Max value = {maxvalue}. Do you intended ?")
        
        # preprocess for pretrained models
        x = image_tensor / 2.0 + 0.5 # [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(x))
        return dataset
