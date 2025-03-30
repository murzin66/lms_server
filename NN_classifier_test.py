import torch
import torch.nn.functional as F
import joblib
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Определение архитектуры модели GAT
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Загрузка векторайзера
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
model = GAT(in_channels=len(vectorizer.get_feature_names_out()), hidden_channels=8, out_channels=8, heads=8, dropout=0.5).to(device)

# Загрузка весов модели
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

# Словарь для отображения индексов классов в названия
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

# Функция для классификации нового текстового описания
def classify_new_interest(description):
    # Векторизация нового описания
    x_new = vectorizer.transform([description]).toarray()
    x_new = torch.tensor(x_new, dtype=torch.float).to(device)

    # Создание пустого edge_index, так как у нас нет связей для нового узла
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    # Предсказание класса
    with torch.no_grad():
        out = model(x_new, edge_index)
        pred = out.argmax(dim=1).item()
        predicted_class = class_mapping[pred]
        print(f'Предсказанный класс: {predicted_class}')

# Пример использования

new_descriptions = ["Интересуюсь нейронными сетями и искусственным интеллектом",
                    "Изучаю биофизику, исследую физические процессы в живых организмах.",
                    "Создаю веб-приложения с использованием Phoenix Framework и обеспечиваю их масштабируемость.",
                    "Применяю методы машинного обучения для детекции аномалий в данных.",
                    "Изучаю зоологию, исследую поведение и экологию животных в дикой природе."]
for i in new_descriptions:
    (classify_new_interest(i))

#classify_new_interest(new_description)
