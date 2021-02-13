import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from utils import gaussian_kernel, feat


def get_datasets(path, download_=True):
  # Train data

  img_transforms = transforms.Compose([
          transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
          transforms.Resize((384,384)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                              std=(0.229, 0.224, 0.225) )
  ])

  train_ds = VOCDetection(root=path, 
                          transform=img_transforms, 
                          target_transform=feat,
                          image_set="trainval", year="2012", download=download_)

  train_dl =  DataLoader(train_ds, 
                      batch_size=32, 
                      num_workers=0, 
                      shuffle=True)



  # Test data

  img_transforms = transforms.Compose([
          transforms.Resize((384,384)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                              std=(0.229, 0.224, 0.225) )
  ])

  val_ds = VOCDetection(root=path, 
                      transform=img_transforms, 
                      target_transform=feat,
                      image_set="test", year="2007", download=download_)

  val_dl =  DataLoader(val_ds, 
                      batch_size=32, 
                      num_workers=0, 
                      shuffle=True)

  return train_dl, val_dl