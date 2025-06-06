import utils as u
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

test_path = 'data\Testing'
train_path = 'data\Training'
val_path = 'data\Validation'
classes = ["glioma", "meningioma", "pituitary", "notumor"]
class_dict = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

def create_val():
    u.create_validation_split(train_path,val_path)

def visualise():
    tr_df = u.train_df(train_path)
    ts_df = u.train_df(test_path)
    v_df = u.train_df(val_path)

    
    u.visualise(tr_df, 'training')
    u.visualise(ts_df, 'testing')
    u.visualise(v_df, 'validation')

    orig,aug=u.list_images(train_path,classes)
    print("original img count:",len(orig))
    print("augmented img count:",len(aug))



def train(epochs=10, batch_size=32):
    tr_df = u.train_df(train_path)

    tr_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    v_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_path, transform=tr_transform)
    val_dataset = ImageFolder(root=val_path, transform=v_transform)

    class_to_idx = train_dataset.class_to_idx
    label_map = {v: k for k, v in class_to_idx.items()}
    numerical_labels = tr_df['Class'].map(class_to_idx)
    
    class_counts = tr_df['Class'].value_counts()
    weights = 1.0 / class_counts[numerical_labels].values
    sampler = torch.utils.data.WeightedRandomSampler(weights,len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #defined explicitly

    model = torch.hub.load('pytorch/vision','resnet18',weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features,4)
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=0.001)

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for img,lbl in train_loader:
            img,lbl = img.to(device),lbl.to(device)
            opt.zero_grad()
            op = model(img)
            loss = crit(op,lbl)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
    
        # val 
        model.eval()
        val_corr = 0
        val_tot = 0

        with torch.no_grad():
            for img,lbl in val_loader:
                img,lbl = img.to(device),lbl.to(device)
                op = model(img)
                _,pred = torch.max(op,1)
                val_tot += lbl.size(0)
                val_corr += (pred == lbl).sum().item()

        val_acc = 100 * val_corr / val_tot if val_tot > 0 else 0.0 #avoiding div by 0
        print(f"Epoch {ep+1}/{epochs}, Train Loss: {tr_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.2f}%")    
    torch.save(model.state_dict(), 'resnet18_brain_tumor.pth')

def test_model(batch_size=32):
    v_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageFolder(root=test_path, transform=v_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision', 'resnet18', weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load('resnet18_brain_tumor.pth'))
    model = model.to(device)
    model.eval()
    test_corr = 0
    test_tot = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            op = model(img)
            _, pred = torch.max(op, 1)
            test_tot += lbl.size(0)
            test_corr += (pred == lbl).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    test_acc = 100 * test_corr / test_tot if test_tot > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return test_acc


if __name__=='__main__':
    # create_val() # running once is enough
    # visualise()
    # train()
    test_model()