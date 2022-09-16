import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets

from config import cfg
from model import NICE
import torchvision
# Data
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)

model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])
if cfg['USE_CUDA']:
  device = torch.device('cuda')
  model = model.to(device)

# Train the model
model.train()

opt = optim.Adam(model.parameters())

for i in range(cfg['TRAIN_EPOCHS']):
    mean_likelihood = 0.0
    num_minibatches = 0

    for batch_id, (x, _) in enumerate(dataloader):
        x = x.view(-1, 784) + torch.rand(784) / 256.
        if cfg['USE_CUDA']:
            x = x.cuda()

        x = torch.clamp(x, 0, 1)

        z, likelihood = model(x)
        loss = -torch.mean(likelihood)   # NLL

        loss.backward()
        opt.step()
        model.zero_grad()

        mean_likelihood -= loss
        num_minibatches += 1

    mean_likelihood /= num_minibatches
    print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))

    if i % 5 == 0:
        save_path = os.path.join(cfg['MODEL_SAVE_PATH'], '{}.pt'.format(i))
        torch.save(model, save_path)
        sample_img=model.sample(16)
        sample_img.clamp_(0,255)
        smaple_img=sample_img.view(16,1,28,28)
        print(smaple_img.shape)
        torchvision.utils.save_image(sample_img,f"/home/whq/data/code/NICE/output/epoch{i}.jpg",nrow=4,padding=2,normalize=False)



