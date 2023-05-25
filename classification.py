import torch

def get_classification_predictions(model, loader):
  model.eval()

  with torch.no_grad():
    for loader_ in loader:
      for data in loader_:
        output = model(data)
        _, predictions = torch.max(output, dim=1)
    
    return predictions