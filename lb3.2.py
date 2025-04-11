# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:26:12 2025

@author: bfyd
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

print(df.head())

X = df.iloc[:, :4].values  
y = df.iloc[:, 4].values   

label_map = {"Iris-setosa": 0, "Iris-versicolor": 1}
y = np.array([label_map[label] for label in y])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

model = nn.Linear(4, 2)

loss_fn = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(100):
    outputs = model(X_tensor)
    
    loss = loss_fn(outputs, y_tensor)
    
    optimizer.zero_grad()  
    loss.backward()        
    optimizer.step()      
    
    if (i + 1) % 10 == 0:
        print(f'{i+1}/{100}, Ошибка: {loss.item():.4f}')

with torch.no_grad():  
    predictions = model(X_tensor)
    _, predicted_classes = torch.max(predictions, 1)  

t = torch.tensor(y, dtype=torch.long)  

print("Предсказанные классы:", predicted_classes)
print("Эталонные метки:", t)