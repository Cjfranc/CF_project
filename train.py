import torch

from model import CenterNet
from data import get_datasets
from losses import L1_loss, focal_loss

#train and validation loaders
train_dl, val_dl = get_datasets("/home/student/CenterNet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CenterNet model
m = CenterNet(20)
m = m.to(device)

opt = torch.optim.Adam(m.parameters(), lr=0.000125)
#learning rate decay after given number of epochs
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[45, 60], gamma=0.1)

train_list = []
val_list = []



for epoch in range(70):
  print("Epoch: ", epoch+1)
  
  if (epoch + 1) % 10 == 0:
    torch.save(m.state_dict(), "./model_epoch_"+str(epoch)+".pth")

  # Train
  loss_ = 0
  m.train()

  #y ground-truth labels determined from feat() in utils
  for x, (y_hm, y_size) in train_dl:
    x = x.to(device)
    y_hm = y_hm.to(device) #target values found using gaus kernel
    y_size = y_size.to(device)

    opt.zero_grad()
    y_pred = m(x)
    
    l = focal_loss(y_hm, y_pred[0]) + 0.02*L1_loss(y_size, y_pred[1])
    loss_ += l.item()/len(train_dl)

    l.backward()
    opt.step()
    
  scheduler.step()

  train_list.append(loss_)
  print("Train loss:", loss_)
  
  # Evaluation
  loss_ = 0
  m.eval()
  
  with torch.no_grad():
    for x, (y_hm, y_size) in val_dl:
      x = x.to(device)
      y_hm = y_hm.to(device)
      y_size = y_size.to(device)

      y_pred = m(x)
      
      l = focal_loss(y_hm, y_pred[0]) + 0.02*L1_loss(y_size, y_pred[1])
      loss_ += l.item()/len(val_dl)

  val_list.append(loss_)
  print("Val loss:", loss_)


with open("train_list.txt", "w") as f:
  for row in train_list:
    f.write(str(row) + '\n')


with open("val_list.txt", "w") as f:
  for row in val_list:
    f.write(str(row) + '\n')