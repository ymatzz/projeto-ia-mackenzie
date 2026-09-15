# ==============================================================================
# UNIVERSIDADE PRESBITERIANA MACKENZIE
# Faculdade de Computação e Informática - FCI
# Disciplina: Inteligência Artificial (7ºJ SI - Noite)
# Professor: Prof. Dr. Leandro Zerbinatti
# ------------------------------------------------------------------------------
# PROJETO: Análise Preditiva com IA para Exercícios de Musculação
# ARQUIVO: analise_exploratoria_uiprmd.py
# ------------------------------------------------------------------------------
# INTEGRANTES DO GRUPO:
# - Bruna França Martinez (RA: 10420225) - 10420225@mackenzista.com.br
# - Graziely de Oliveira Severo (RA: 10425431) - 10425431@mackenzista.com.br
# - Matheus Fernandes dos Santos (RA: 10420439) - 10420439@mackenzista.com.br
# ------------------------------------------------------------------------------
# SÍNTESE DO CONTEÚDO:
# Este script realiza a Análise Exploratória de Dados (EDA) e pré-processamento
# da base cinemática de agachamento profundo (UI-PRMD / Deep Squat), validando
# a pipeline de dados do projeto. O código inclui carregamento, estatísticas
# descritivas, comparação de trajetórias temporais de ângulos articulares entre
# execuções corretas e incorretas, análise do espaço latente (Autoencoder)
# e avaliação dos escores de qualidade biomecânica (Quality Scores via GMM).
# ------------------------------------------------------------------------------
# HISTÓRICO DE ALTERAÇÕES:
# Data        | Autor                      | Breve Descrição da Atualização
# ------------|----------------------------|------------------------------------
# 14/09/2026  | Matheus Fernandes Santos   | Criação do script e carregamento da base dados
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuração de estilo visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('Set2')

def carregar_dados():
    """
    Carrega os arquivos CSV do dataset UI-PRMD (Deep Squat)
    """
    print("=== 1. CARREGAMENTO DOS DADOS ===")
    data_corr = pd.read_csv('dataset/Data_Correct.csv', header=None)
    data_inc = pd.read_csv('dataset/Data_Incorrect.csv', header=None)
    
    lbl_corr = pd.read_csv('dataset/Labels_Correct.csv', header=None, names=['quality_score'])
    lbl_inc = pd.read_csv('dataset/Labels_Incorrect.csv', header=None, names=['quality_score'])
    
    ae_corr = pd.read_csv('dataset/Autoencoder_Output_Correct.csv', header=None)
    ae_inc = pd.read_csv('dataset/Autoencoder_Output_Incorrect.csv', header=None)
    
    print(f"Data Correct (Ângulos): {data_corr.shape}")
    print(f"Data Incorrect (Ângulos): {data_inc.shape}")
    print(f"Labels Correct: {lbl_corr.shape}")
    print(f"Labels Incorrect: {lbl_inc.shape}")
    print(f"Autoencoder Correct (Espaço Latente): {ae_corr.shape}")
    print(f"Autoencoder Incorrect (Espaço Latente): {ae_inc.shape}")
    
    return data_corr, data_inc, lbl_corr, lbl_inc, ae_corr, ae_inc

def analise_estatistica_labels(lbl_corr, lbl_inc):
    """
    Gera estatísticas descritivas dos Quality Scores e plota gráfico comparativo
    """
    print("\n=== 2. ESTATÍSTICA DESCRITIVA DOS ESCORES DE QUALIDADE ===")
    lbl_corr['classe'] = 'Correto'
    lbl_inc['classe'] = 'Incorreto'
    df_labels = pd.concat([lbl_corr, lbl_inc], ignore_index=True)
    
    estatisticas = df_labels.groupby('classe')['quality_score'].describe()
    print(estatisticas)
    
    # Plot Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='classe', y='quality_score', data=df_labels, palette=['#2ecc71', '#e74c3c'])
    plt.title('Distribuição do Escore de Qualidade de Movimento (GMM Quality Score)', fontsize=13, fontweight='bold')
    plt.xlabel('Classe da Execução Técnica', fontsize=11)
    plt.ylabel('Escore de Qualidade (0 a 1)', fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_boxplot_quality_scores.png', dpi=300)
    plt.show()
    
    return df_labels

def analise_trajetorias_temporais(data_corr, data_inc):
    """
    Compara o comportamento temporal dos ângulos nas execuções corretas e incorretas
    """
    print("\n=== 3. ANÁLISE DE TRAJETÓRIAS TEMPORAIS DOS ÂNGULOS ARTICULARES ===")
    
    plt.figure(figsize=(10, 5))
    frames = np.arange(240)
    
    # Plot das 5 primeiras repetições do Ângulo 0
    for i in range(5):
        plt.plot(frames, data_corr.iloc[i], color='#2ecc71', alpha=0.7, label='Correto' if i == 0 else "")
        plt.plot(frames, data_inc.iloc[i], color='#e74c3c', alpha=0.7, label='Incorreto' if i == 0 else "")
        
    plt.title('Comportamento Angular Temporal (Ângulo Articular 0 - Primeiras Execuções)', fontsize=13, fontweight='bold')
    plt.xlabel('Frame Temporal (0 a 239)', fontsize=11)
    plt.ylabel('Ângulo Normalizado / Centrado', fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('eda_trajetoria_angular.png', dpi=300)
    plt.show()

def analise_espaco_latente_autoencoder(ae_corr, ae_inc, lbl_corr, lbl_inc):
    """
    Avalia a dispersão das variáveis do Autoencoder e testa um modelo baseline de Regressão Linear
    """
    print("\n=== 4. MODELO BASELINE DE REGRESSÃO LINEAR NO ESPAÇO LATENTE ===")
    
    X = pd.concat([ae_corr, ae_inc], ignore_index=True)
    y = pd.concat([lbl_corr['quality_score'], lbl_inc['quality_score']], ignore_index=True)
    
    # Treino de Regressão Linear
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot das Predições
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, alpha=0.7, color='#3498db')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title('Regressão Linear Baseline: Escore Real vs. Escore Predito', fontsize=13, fontweight='bold')
    plt.xlabel('Escore Real (GMM)', fontsize=11)
    plt.ylabel('Escore Predito (Regressão Linear)', fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_regressao_baseline.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    data_corr, data_inc, lbl_corr, lbl_inc, ae_corr, ae_inc = carregar_dados()
    df_labels = analise_estatistica_labels(lbl_corr, lbl_inc)
    analise_trajetorias_temporais(data_corr, data_inc)
    analise_espaco_latente_autoencoder(ae_corr, ae_inc, lbl_corr, lbl_inc)
    print("\nAnálise exploratória executada com sucesso!")