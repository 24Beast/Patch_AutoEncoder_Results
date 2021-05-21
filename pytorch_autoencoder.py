# Importing Libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Creating Random Samples
num_iter = 1000
fname = "./orig_0.png"
fout = "./"


orig = cv2.imread(fname)/255

w,h,c = orig.shape

# Params generated, not saving actual image samples
w_iter = np.random.randint(low = 10, high = w, size = num_iter)
h_iter = np.random.randint(low = 10, high = h, size = num_iter)
x_iter = np.random.randint(low = 0, high = w-w_iter, size = num_iter)
y_iter = np.random.randint(low = 0, high = h-h_iter, size = num_iter)

for i in range(int(num_iter/100)):
    print("\rWorking on iter : "+str(i),end = "")
    x0,y0,w0,h0 = x_iter[i],y_iter[i],w_iter[i],h_iter[i]
    cv2_imshow(orig[x0:x0+w0,y0:y0+h0]*255)
#    cv2.imwrite(fout+"RandomSample_"+str(i)+".png",orig[x0:x0+w0,y0:y0+h0])

# Utility Functions

def show_cuda_img(img,fname):
  new_img = img.cpu().detach().numpy()[0].copy().transpose((1, 2, 0))
  cv2_imshow(new_img*255)
  cv2.imwrite(fname+".png",new_img*255)

# Model Definition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=2)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=2)
        self.conv3 = nn.Conv2d(64, 256, 3, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        z = x.size()
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv3(x))
        x = x[:z[0],:z[1],:z[2],:z[3]]
        return x

model = AutoEncoder()
model.to(device)
print(model)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
output_dir = "../gdrive/MyDrive/Experiments/AutoEncoder/"
n_epochs = 100
train_ratio = 0.8
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for i in range(int(num_iter*train_ratio)):
        x0,y0,w0,h0 = x_iter[i],y_iter[i],w_iter[i],h_iter[i]
        img = np.expand_dims(orig[x0:x0+w0,y0:y0+h0].copy().transpose(2,0,1),axis=0)
        img = torch.Tensor(img).to(device)
        optimizer.zero_grad()
        outputs = model(img)
#        outputs = outputs[:,:,:w0,:h0]
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/int(num_iter*train_ratio)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    if(epoch%10==0):
      x1,y1,w1,h1 = x_iter[0],y_iter[0],w_iter[0],h_iter[0]
      img = torch.Tensor(np.expand_dims(orig[x1:x1+w1,y1:y1+h1].copy().transpose(2,0,1),axis=0)).to(device)
      outputs = model(img)
      show_cuda_img(img,output_dir + "Train/Img_"+str(epoch))
      show_cuda_img(outputs,output_dir + "Train/Out_"+str(epoch))

# Test Outputs
for i in range(int(0.8*num_iter),num_iter):
  x1,y1,w1,h1 = x_iter[i],y_iter[i],w_iter[i],h_iter[i]
  img = torch.Tensor(np.expand_dims(orig[x1:x1+w1,y1:y1+h1].copy().transpose(2,0,1),axis=0)).to(device)
  output = model(img)
  show_cuda_img(torch.cat((img,output),axis=2),output_dir+"Test/"+str(i))
  print("\n")