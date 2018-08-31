import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_epoches = 66
learning_rate = 0.001

transform = transforms.Compose([
     transforms.Pad(4),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32),
     transforms.ToTensor()
]
)          
                                          
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transform,
                                             download=False)
test_dataset= torchvision.datasets.CIFAR10(root='data/',
                                              train=False,
                                              transform=transform)  

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=64,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=64,
                                         shuffle=False)                                         
                                                                                                                                       
class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,
                               stride = stride,padding=1,bias=False)   

        # self.diconv=nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     # stride=1, padding=2, bias=False,dilation=2)       
        
        #bn1 and bn2 --->bn?  no!!!!             
        self.bn1=nn.BatchNorm2d(out_channels)
        
        self.relu=nn.ReLU(inplace=True)
        
        self.diconv=nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False) 
        self.bn2=nn.BatchNorm2d(out_channels)             
                     
        self.downsample=downsample
        
    def forward(self,x):
        # print('--------------------------------------------------------------')
        # print('x:',x.size())
        residual=x       
        out=self.conv(x)
        out=self.bn1(out)
        out=self.relu(out)   
     
        out=self.diconv(out)
        out=self.bn2(out)
        
        if self.downsample:
            residual=self.downsample(x) 
        # print(self.downsample)
        out+=residual
        out=self.relu(out)    
        return out
       
class ResNet(nn.Module):
    def __init__(self, residualblock,layers, num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels=16
        
        self.conv=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(16)
        self.relu= nn.ReLU(inplace=True)
        self.layer1=self.make_layer(residualblock,16,layers[0])
        self.layer2=self.make_layer(residualblock,32,layers[1],2)
        self.layer3=self.make_layer(residualblock,64,layers[2],2)
        self.avg_pool=nn.AvgPool2d(8)
        self.fc=nn.Linear(64,num_classes)
        
    def make_layer(self,residualblock,out_channels, num_blocks, stride=1):
        downsample=None
        if(stride!=1 or (self.in_channels!=out_channels)):
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels, kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm2d(out_channels)
            )
            
        layers=[]
        layers.append(residualblock(self.in_channels,out_channels,stride, downsample))
        self.in_channels=out_channels
        for i in range(1,num_blocks):
            layers.append(residualblock(out_channels,out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self,x):
        #print(x.size())
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.layer1(out)
        #'layer1:', (64, 16, 32, 32)
        out=self.layer2(out)
        #'layer2: ', (64, 32, 16, 16)
        out=self.layer3(out)
        #'layer3: ', (64, 64, 8, 8))
        out=self.avg_pool(out)#------>(64, 64, 1, 1)       
        out=out.view(out.size(0),-1)#--------->(64, 64)
        out=self.fc(out)
        return out
                    
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(total_epoches):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, total_epoches, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
        
        
    # Test the model
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))    

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
