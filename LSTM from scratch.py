#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt


# In[2]:


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def tanh(x):
    return math.tanh(x)


# In[9]:


def lstm_predict(x):
    Wf, Whf, bf = 0.5, 0.1, 0
    Wi, Whi, bi = 0.6, 0.2, 0
    Wc, Whc, bc = 0.7, 0.3, 0
    Wo, Who, bo = 0.8, 0.4, 0

    h, C = 0, 0  
    losses = []

    for t in range(len(x)):
        xt = x[t]

        ft = sigmoid(Wf * xt + Whf * h + bf)
        it = sigmoid(Wi * xt + Whi * h + bi)
        C_candidate = tanh(Wc * xt + Whc * h + bc)
        C = ft * C + it * C_candidate
        ot = sigmoid(Wo * xt + Who * h + bo)
        h = ot * tanh(C)

        loss = (h - xt) ** 2  
        losses.append(loss)

        print(f"Time Step {t+1}: ft={ft:.4f}, it={it:.4f}, C_candidate={C_candidate:.4f}, C={C:.4f}, ot={ot:.4f}, h={h:.4f}, loss={loss:.4f}")
    
    predicted_value = h  
    print(f"Predicted next value: {predicted_value:.4f}")

    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title('Loss over Time Steps')
    plt.show()

    return predicted_value


# In[10]:


x = [1, 2, 3]
next_value = lstm_predict(x)

