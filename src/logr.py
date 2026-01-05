import numpy as np
import matplotlib.pyplot as plt

# data
np.random.seed(42)
n = 100  # persons

age = np.random.randint(30, 80, size=n)  # age generate kro

risk_score = age - 50 + np.random.randn(n) * 5  # adding noise

y = (risk_score > 0).astype(int)  # binary conversion

X = age

# sigmoid use krege
def sigmoid(z):
    return 1 / (1 + np.exp(z))




# log loss fun used
def log_l(y, p):
    epsilon = 1e-15  # log 0 ko avoid krta h
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# gradient decent but different initilize m,b
m = 0.0
b = 0.0


# repeat hoga
def train_logistic(X, y, learning_rate=0.0001, iterations=1000):
    global m, b  # m aur b ko update krne ke liye

    for i in range(iterations):

        # probability
        z = m * X + b
        p = sigmoid(z)

        # gradients
        dm = np.mean((p - y) * X)  # model agr jhya high jaye +ve weight kam
        db = np.mean(p - y)        # model -ve m jaye weight jhyada

        # parameters ko update kro
        m -= learning_rate * dm
        b -= learning_rate * db


train_logistic(X, y, learning_rate=0.0001, iterations=1000)#training krr raha

z = m * X + b
pro = sigmoid(z)
y_pred_class = (pro >= 0.5).astype(int) #probability convert kiya

# final outputs
print("Final weight (m):", m)
print("Final bias (b):", b)
print("Final Log Loss:", log_l(y, pro))

# visualization last dekho
plt.scatter(X, y, label="Actual Data")
plt.scatter(X, pro, label="Predicted Probability")
plt.xlabel("Age")
plt.ylabel("Probability / Class")
plt.legend()
plt.show()