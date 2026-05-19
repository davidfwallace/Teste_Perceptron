import numpy as np
import matplotlib.pyplot as plt

# --- PASSO 1: Carregar e Filtrar Dados ---
def carregar_dados(caminho_arquivo):
    dados = np.loadtxt(caminho_arquivo)
    filtro = (dados[:, 0] == 1) | (dados[:, 0] == 5)
    dados_filtrados = dados[filtro]
    
    X = dados_filtrados[:, 1:3] # Colunas de intensidade e simetria
    y_original = dados_filtrados[:, 0]
    y = np.where(y_original == 1, 1, -1) # 1 vira 1, e 5 vira -1
    
    return X, y

# --- PASSO 2: O Algoritmo do Perceptron ---
def treinar_perceptron(X, y, taxa_aprendizado=0.1, epocas=100):
    n_amostras, n_caracteristicas = X.shape
    pesos = np.zeros(n_caracteristicas)
    bias = 0.0
    
    for epoca in range(epocas):
        erros_na_epoca = 0
        for i in range(n_amostras):
            z = np.dot(X[i], pesos) + bias
            y_previsto = 1 if z >= 0 else -1
            
            if y[i] != y_previsto:
                erro = y[i] - y_previsto 
                pesos += taxa_aprendizado * erro * X[i]
                bias += taxa_aprendizado * erro
                erros_na_epoca += 1
                
        if erros_na_epoca == 0:
            print(f"-> Treinamento concluído! Convergiu na época {epoca+1}.")
            break
            
    return pesos, bias

# --- PASSO 3: Desenhar o Gráfico ---
def plotar_grafico(X, y, pesos, bias):
    # Separa os pontos para o gráfico baseado no rótulo
    X_digito_1 = X[y == 1]
    X_digito_5 = X[y == -1]

    plt.figure(figsize=(10, 6))
    
    # Desenha os pontos espalhados (Scatter Plot)
    plt.scatter(X_digito_1[:, 0], X_digito_1[:, 1], color='blue', label='Dígito 1', alpha=0.5)
    plt.scatter(X_digito_5[:, 0], X_digito_5[:, 1], color='red', label='Dígito 5', alpha=0.5)

    # A matemática para desenhar a reta: w1*x1 + w2*x2 + b = 0
    # Isolamos o x2 para achar os pontos da reta no gráfico
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    
    # Evita divisão por zero caso o peso 2 seja muito pequeno
    if pesos[1] != 0: 
        x2_min = (-pesos[0] * x1_min - bias) / pesos[1]
        x2_max = (-pesos[0] * x1_max - bias) / pesos[1]
        plt.plot([x1_min, x1_max], [x2_min, x2_max], color='black', linewidth=2, label='Reta do Perceptron')

    plt.title('Perceptron: Dígito 1 (Azul) vs Dígito 5 (Vermelho)')
    plt.xlabel('Característica 1: Intensidade')
    plt.ylabel('Característica 2: Simetria')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# EXECUTANDO TUDO
# ==========================================
print("Carregando dados...")
X_treino, y_treino = carregar_dados('digits.train')

print("Treinando o Perceptron...")
pesos_finais, bias_final = treinar_perceptron(X_treino, y_treino, taxa_aprendizado=0.1, epocas=100)

print("\n--- Resultados ---")
print(f"Peso Intensidade: {pesos_finais[0]:.4f}")
print(f"Peso Simetria: {pesos_finais[1]:.4f}")
print(f"Bias: {bias_final:.4f}")

print("Gerando gráfico...")
plotar_grafico(X_treino, y_treino, pesos_finais, bias_final)