import torch

def focal_loss(y, y_pred):
  pos = y.eq(1).float()
  neg = y.lt(1).float()

  y_pred = torch.clamp(y_pred, 1e-6, 1-1e-6)

  pos_loss = torch.pow(1-y_pred, 2)*torch.log(y_pred)*pos
  neg_loss = torch.pow(1-y, 4)*torch.pow(y_pred, 2)*torch.log(1-y_pred)*neg
  
  N = pos.sum()
  neg_loss = neg_loss.sum()
  pos_loss = pos_loss.sum()
  
  if N < 1:
    loss = neg_loss
  else:
    loss = (neg_loss + pos_loss) / N
  
  return -loss


def L1_loss(y, y_pred):
  kp = y.gt(0).float()
  N = kp[0].sum()

  loss = (torch.abs(y-y_pred)*kp)/N
  
  return loss.sum()