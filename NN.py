import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import joblib

# 1. Загрузка тестовых данных
# Предположим, что у вас есть DataFrame с тестовыми данными
df_test = pd.read_csv('new_students_interests.csv')  # Замените на путь к вашему CSV-файлу
df_test.dropna(subset=['Описание интересов студента'], inplace=True)

# 2. Загрузка сохранённого векторизатора
vectorizer = joblib.load('vectorizer.pkl')

# 3. Векторизация тестовых данных
X_test = vectorizer.transform(df_test['Описание интересов студента']).toarray()

# 4. Построение графа на основе косинусного сходства
cosine_sim = cosine_similarity(X_test, X_test)
edges = []
threshold = 0.3  # Пониженный порог схожести
n = len(df_test)
for i in range(n):
    for j in range(i + 1, n):
        if cosine_sim[i, j] > threshold:
            edges.append([i, j])

# Check if no edges are found
if len(edges) == 0:
    print("No edges found based on cosine similarity.")
    # Handle the case where no edges are found, e.g., fallback or skip prediction
else:
    print(f"Found {len(edges)} edges.")

edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 5. Создание объекта Data для PyTorch Geometric
x = torch.tensor(X_test, dtype=torch.float)
edge_index = edges if edges.size(1) > 0 else torch.empty(2, 0, dtype=torch.long)  # Fallback if no edges
data = Data(x=x, edge_index=edge_index)

# 6. Определение модели GAT
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

# 7. Инициализация модели и загрузка сохранённых весов
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(in_channels=X_test.shape[1], hidden_channels=8, out_channels=8, heads=8, dropout=0.5).to(device)
model.load_state_dict(torch.load('model_weights.pth'), strict=False)
model.eval()

# 8. Предсказание классов для тестовых данных
data = data.to(device)
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)

class_mapping = {
    0: 'DevOps',
    1: 'Web',
    2: 'ML',
    3: 'Software',
    4: 'Art',
    5: 'Biology',
    6: 'Physics',
    7: 'Math'
}
predicted_classes = [class_mapping[int(label)] for label in pred]

# 9. Вывод предсказанных классов
print(predicted_classes)
