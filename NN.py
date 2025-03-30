import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import joblib

# 1. Загрузка и предобработка данных
df = pd.read_csv('students_interests.csv')  # Замените на путь к вашему CSV-файлу
df.dropna(subset=['Описание интересов студента'], inplace=True)

# Токенизация и векторизация интересов студентов
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Описание интересов студента']).toarray()

# Сохранение параметров векторайзера
vectorizer_params = vectorizer.get_params()

# Преобразование классов в числовой формат
classes = df['Класс'].unique()
class_map = {cls: idx for idx, cls in enumerate(classes)}
y = np.array([class_map[cls] for cls in df['Класс']])

# 2. Построение графа на основе косинусного сходства
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X, X)
edges = []
threshold = 0.5  # Порог схожести
n = len(df)
for i in range(n):
    for j in range(i + 1, n):
        if cosine_sim[i, j] > threshold:
            edges.append([i, j])
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 3. Создание объекта Data для PyTorch Geometric
x = torch.tensor(X, dtype=torch.float)
edge_index = edges
y = torch.tensor(y, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)

# Разделение данных на обучающую и тестовую выборки
num_nodes = len(y)
indices = np.arange(num_nodes)
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True
data.train_mask = train_mask
data.test_mask = test_mask

# 4. Определение модели GAT
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 5. Инициализация модели, оптимизатора и функции потерь
model = GAT(in_channels=X.shape[1], hidden_channels=8, out_channels=len(classes), heads=8, dropout=0.5).to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.NLLLoss()

# 6. Функции для обучения и оценки модели
def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct_train = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
    correct_test = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    train_acc = correct_train / data.train_mask.sum().item()
    test_acc = correct_test / data.test_mask.sum().item()
    return train_acc, test_acc

# 7. Обучение модели
epochs = 80
for epoch in range(epochs):
    loss = train_epoch(model, data, optimizer, criterion)
    train_acc, test_acc = evaluate(model, data)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 8. Сохранение параметров модели и параметров векторайзера
torch.save(model.state_dict(), 'model_weights.pth')
joblib.dump(vectorizer_params, 'vectorizer_params.pkl')