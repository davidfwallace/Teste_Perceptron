import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── PASSO 1: Carregar e Filtrar Dados ─────────────────────────────────────────
def carregar_dados(caminho_arquivo):
    dados = np.loadtxt(caminho_arquivo)
    filtro = (dados[:, 0] == 1) | (dados[:, 0] == 5)
    dados_filtrados = dados[filtro]

    X = dados_filtrados[:, 1:3]           # intensidade e simetria
    y = np.where(dados_filtrados[:, 0] == 1, 1, -1)
    return X, y

# ── PASSO 2: Pocket PLA (Atualizado para 2 históricos) ────────────────────────
def treinar_pocket(X, y, taxa_aprendizado=0.01, epocas=1000):
    n, d = X.shape
    pesos = np.zeros(d)
    bias  = 0.0

    def calc_erro(p, b):
        pred = np.where(np.dot(X, p) + b >= 0, 1, -1)
        return (pred != y).sum()

    pocket_pesos = pesos.copy()
    pocket_bias  = bias
    melhor_erro  = calc_erro(pesos, bias)
    
    historico_pla = []
    historico_pocket = []

    for epoca in range(epocas):
        erros_epoca = 0
        indices = np.random.permutation(n)

        for i in indices:
            z      = np.dot(X[i], pesos) + bias
            y_prev = 1 if z >= 0 else -1

            if y[i] != y_prev:
                pesos += taxa_aprendizado * y[i] * X[i]
                bias  += taxa_aprendizado * y[i]
                erros_epoca += 1

        # Fim da época: calcula o erro do PLA (que pode ter piorado)
        erro_atual_pla = calc_erro(pesos, bias)
        historico_pla.append(erro_atual_pla / n)

        # Atualiza o Pocket se o PLA atual for ESTRITAMENTE melhor
        if erro_atual_pla < melhor_erro:
            melhor_erro  = erro_atual_pla
            pocket_pesos = pesos.copy()
            pocket_bias  = bias

        historico_pocket.append(melhor_erro / n)

        if erros_epoca == 0:
            print(f"  Convergiu na época {epoca + 1}!")
            break

    return pocket_pesos, pocket_bias, np.array(historico_pla), np.array(historico_pocket)

# ── PASSO 3: Avaliação (Mantido igual) ────────────────────────────────────────
def avaliar(X, y, pesos, bias, nome=""):
    pred = np.where(np.dot(X, pesos) + bias >= 0, 1, -1)
    acc  = (pred == y).mean()
    erro = 1 - acc
    print(f"  [{nome}]  Acurácia: {acc * 100:.2f}%  |  Erro: {erro * 100:.2f}%")
    return erro

# ── PASSO 4: Gráficos (Atualizado) ────────────────────────────────────────────
def plotar_fronteira(X, y, pesos, bias, titulo="Conjunto"):
    X1 = X[y ==  1]
    X5 = X[y == -1]

    plt.scatter(X1[:, 0], X1[:, 1], color='#2563EB', label='Dígito 1', alpha=0.6, s=25)
    plt.scatter(X5[:, 0], X5[:, 1], color='#DC2626', label='Dígito 5', alpha=0.6, s=25)

    if abs(pesos[1]) > 1e-10:
        x0_min = X[:, 0].min() - 0.05
        x0_max = X[:, 0].max() + 0.05
        x1_min = (-pesos[0] * x0_min - bias) / pesos[1]
        x1_max = (-pesos[0] * x0_max - bias) / pesos[1]
        plt.plot([x0_min, x0_max], [x1_min, x1_max],
                 color='black', linewidth=2, label='Fronteira de decisão')

    plt.title(f'Pocket PLA — {titulo}', fontsize=13, fontweight='bold')
    plt.xlabel('Característica 1: Intensidade')
    plt.ylabel('Característica 2: Simetria')
    plt.legend()
    plt.grid(True, alpha=0.3)

def plotar_curva(historico_pla, historico_pocket):
    # Vamos dar "zoom" nas primeiras 100 épocas para a curva não ficar um traço espremido
    limite = min(len(historico_pla), 100) 
    
    plt.plot(historico_pla[:limite], color='gray', alpha=0.4, lw=1.5, label='PLA (Erro flutuante)')
    plt.step(range(limite), historico_pocket[:limite], color='#2563EB', lw=2.5, where='post', label='Pocket (Melhor salvo)')
    
    plt.xlabel('Época (Zoom nas primeiras 100)')
    plt.ylabel('Taxa de erro (treino)')
    plt.title('Por que usar o Pocket PLA?', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)


# ── EXECUÇÃO (Atualizado) ─────────────────────────────────────────────────────
print("Carregando dados...")
X_treino, y_treino = carregar_dados('digits.train')
X_teste,  y_teste  = carregar_dados('digits.test')
print(f"  Treino: {X_treino.shape[0]} amostras  |  Teste: {X_teste.shape[0]} amostras")

print("\nTreinando Pocket PLA...")
pesos, bias, hist_pla, hist_pocket = treinar_pocket(X_treino, y_treino, taxa_aprendizado=0.01, epocas=1000)

print("\n--- Pesos finais ---")
print(f"  Peso Intensidade : {pesos[0]:.4f}")
print(f"  Peso Simetria    : {pesos[1]:.4f}")
print(f"  Bias             : {bias:.4f}")

print("\n--- Resultados ---")
e_in  = avaliar(X_treino, y_treino, pesos, bias, nome="Treino (E_in) ")
e_out = avaliar(X_teste,  y_teste,  pesos, bias, nome="Teste  (E_out)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

plt.sca(axes[0])
plotar_fronteira(X_treino, y_treino, pesos, bias, titulo=f"Treino  (E_in={e_in*100:.1f}%)")

plt.sca(axes[1])
plotar_fronteira(X_teste, y_teste, pesos, bias, titulo=f"Teste  (E_out={e_out*100:.1f}%)")

plt.sca(axes[2])
plotar_curva(hist_pla, hist_pocket)

plt.suptitle('Perceptron — Dígitos 1 vs 5', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('resultado_perceptron.png', dpi=150, bbox_inches='tight')
plt.show()