import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np

# load data
def get_data():
    data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    train_dataset = datasets.MNIST(root='dataset/',train=False,transform=data_tf,download=True)
    test_dataset = datasets.MNIST(root='dataset/',train=False,transform=data_tf,download=True)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    return train_loader,test_dataset

# build the model
class batch_net(nn.Module):
    def __init__(self,in_dim,hidden1_dim,hidden2_dim,out_dim):
        super(batch_net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(hidden2_dim),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden2_dim,out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



if __name__ == "__main__":
    #hyper parameters
    batch_size = 64
    learning_rate = 1e-2
    num_epoches = 5

    # load data
    train_dataset,test_dataset = get_data()

    # build the model,loss and opt
    model = batch_net(28*28,300,100,10)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)

    # begin training
    step = 0
    for i in range(num_epoches):
        for img,label in train_dataset:
            img,label = img.cuda(),label.cuda()
            img = img.view(64,-1)
            img = Variable(img)
            step += 1
            # print(img.size())
            label = Variable(label)

            # forward
            out = model(img)
            loss = criterion(out,label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 ==0:
                print("steps = {}, loss is {}".format(step,loss))

    # test
    model.eval()
    count =0
    for data in test_dataset:
        img,label = data
        img = img.cuda()
        #img,label = img.cuda(),label.cuda()
        img = img.view(img.size(0),-1)
        img = Variable(img,volatile=True)

        out = model(img)
        _,predict = torch.max(out,1)
        if predict == label:
            count += 1

    print("acc = {} ".format(count/len(test_dataset)))