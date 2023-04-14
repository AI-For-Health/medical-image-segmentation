# train kvasir-seg dataset with resunet
import os
import torch
from image_segmenters import ResUnet
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torch
from torch import nn
from tqdm.auto import tqdm

# define a image segmentation dataset class
class ImageSegmentationDataset:
    # _init_ method
    def __init__(self, path):
        self.path=path
        self.N=len(os.listdir(os.path.join(path, "images")))
        self.names=os.listdir(os.path.join(path, "images"))

    # __len__ method
    def __len__(self):
        return self.N

    # __getitem__ method
    def __getitem__(self, idx):
        image_path=os.path.join(self.path, "images", self.names[idx])
        mask_path=os.path.join(self.path, "masks", self.names[idx])
        img=to_tensor(Image.open(image_path))
        mask=to_tensor(Image.open(mask_path))
        return img, mask[0].reshape(1, mask.shape[1], mask.shape[2])

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target):
        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = inputs.view(-1)
        truth = target.view(-1)
        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()
        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )
        return bce_loss + (1 - dice_coef)

def main():
    dataset_path="data/kvasir-seg-aug"
    N_train=len(os.listdir(os.path.join(dataset_path, "train", "images")))
    N_validation=len(os.listdir(os.path.join(dataset_path, "validation", "images")))

    # load train and validation datasets
    train_dataset=ImageSegmentationDataset(os.path.join(dataset_path, "train"))
    validation_dataset=ImageSegmentationDataset(os.path.join(dataset_path, "validation"))

    # define a model
    model=ResUnet(3)

    # define a loss function
    loss_fn=BCEDiceLoss()

    # define an NAdam optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

    # define a step learning rate scheduler
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # define dataloaders
    train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_dataloader=torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=False)

    # train the model
    for epoch in range(30):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            images, masks=batch
            outputs=model(images)
            loss=loss_fn(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch: %d, Batch: %d, Loss: %f" % (epoch, idx, loss.item()))
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(validation_dataloader):
                images, masks=batch
                outputs=model(images)
                loss=loss_fn(outputs, masks)
                print("Epoch: %d, Batch: %d, Loss: %f" % (epoch, idx, loss.item()))
        lr_scheduler.step()
    torch.save(model.state_dict(), "kvasir-seg-resunet.pt")

    


if __name__ == "__main__":
    main()