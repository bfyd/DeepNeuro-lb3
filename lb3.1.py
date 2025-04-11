# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:42:33 2025

@author: bfyd
"""
import random
import torch
x=torch.zeros(1,3,dtype=torch.int)
print(x)
x = torch.randint(1,10,(1,3))
print(x)
 
x =x.to(dtype=torch.float32)
x.requires_grad=True 
print(x)

y=x**3 
b=random.randint(1,10)
z=y*b
с=torch.exp(z)
out=с.mean()
out.backward()
print(x.grad)
