import ssl, torch
import torchvision.transforms as T
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim

ssl._create_default_https_context = ssl._create_unverified_context

def train_model():
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {d}")
    tr = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.2,0.2), T.ToTensor()])
    vl = T.Compose([T.Resize((224,224)), T.ToTensor()])
    
    tl = DataLoader(datasets.ImageFolder('dataset/train', transform=tr), batch_size=32, shuffle=True)
    v = DataLoader(datasets.ImageFolder('dataset/valid', transform=vl), batch_size=32)
    
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 4)
    m, opt = m.to(d), optim.Adam(m.parameters(), lr=0.001)
    
    b = 0
    t_batches = len(tl)
    
    for epoch in range(5):
        m.train()
        for i, (x, y) in enumerate(tl): 
            opt.zero_grad()
            nn.CrossEntropyLoss()(m(x.to(d)), y.to(d)).backward()
            opt.step()
            
            # Custom Progress Bar
            progress = int(30 * (i + 1) / t_batches)
            print(f"\rEpoch {epoch+1}/5 | [{'='*progress}{' '*(30-progress)}] {i+1}/{t_batches} Batches", end="")
            
        m.eval()
        a = sum(m(x.to(d)).argmax(1).eq(y.to(d)).sum().item() for x,y in v) / len(v.dataset)
        print(f" | Val Acc: {a*100:.2f}%")
        
        if a > b: 
            b = a
            torch.save(m.state_dict(), 'best_model.pth')

train_model()