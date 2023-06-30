from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np


# features (1 = sim , 0 = nao)
# pelo longo?
# perna curta?
# faz au au?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]


treino_x = [porco1, porco2, porco3, cachorro1, cachorro2,  cachorro3]
# f(x) = y
# 1 => porco, 0 => cachorro
treino_y = [1, 1, 1, 0, 0, 0]


model = LinearSVC()
model.fit(treino_x, treino_y)


animal_misterioso = [1, 1, 1]
previsao = model.predict([animal_misterioso])
print(previsao)


m1 = [1, 1, 1]
m2 = [1, 1, 0]
m3 = [0, 1, 1]

teste_x = [m1, m2, m3]
teste_y = [0, 1, 1]

previsao = model.predict(teste_x)

print(previsao)
print((previsao == teste_y))

corretos = (previsao == teste_y)
print(type(corretos))
print(np.sum(corretos))

taxa_acertos = np.sum(corretos)/len(teste_x) * 100
print(f"Taxa de acertos: {taxa_acertos:.2f}%")


acuracia = accuracy_score(teste_y, previsao)
print(f"Acuracia: {acuracia*100:.2f}%")
