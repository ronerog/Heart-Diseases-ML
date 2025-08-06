# Heart-Diseases-ML

## Resumo do Projeto

Este projeto aplica técnicas de Machine Learning para prever a presença de doença cardíaca em pacientes, utilizando o dataset público UCI Heart Disease. O objetivo é consolidar conceitos de análise exploratória, pré-processamento, modelagem, avaliação e comunicação de resultados em um problema real de saúde.

---

## Objetivo

Desenvolver modelos preditivos capazes de identificar pacientes com maior risco de doença cardíaca, auxiliando no diagnóstico clínico e na priorização de exames.

---

## Etapas do Projeto

### 1. Exploração dos Dados
- Identificação da variável-alvo (`num`, convertida para `target` binário: 0 = sem doença, 1 = com doença).
- Análise das principais features: idade, sexo, tipo de dor no peito, pressão arterial, colesterol, glicemia, eletrocardiograma, frequência cardíaca máxima, angina induzida por exercício, depressão do ST, inclinação do ST, número de vasos principais, talassemia.
- Visualização com histogramas, boxplots e countplots.
- Verificação do desbalanceamento de classes.

### 2. Pré-processamento
- Tratamento de valores ausentes: preenchimento de `ca` e `thal` com a moda.
- Conversão de variáveis categóricas com OneHotEncoder.
- Padronização das variáveis numéricas com StandardScaler.

### 3. Divisão dos Dados
- Separação em treino (80%) e teste (20%) com estratificação.
- Garantia de que o pré-processamento foi ajustado apenas nos dados de treino.

### 4. Modelagem e Avaliação
- Treinamento de Regressão Logística e Random Forest (com e sem ajuste de hiperparâmetros via GridSearchCV).
- Avaliação dos modelos com acurácia, precisão, recall, F1-score e matriz de confusão.
- Uso de validação cruzada para maior robustez.
- Visualização dos resultados com gráficos e heatmaps.

### 5. Conclusões e Insights
- Random Forest ajustado apresentou o melhor desempenho geral.
- Variáveis mais relevantes: número de vasos principais (`ca`), tipo de dor no peito (`cp`), frequência cardíaca máxima (`thalach`), depressão do ST (`oldpeak`), inclinação do ST (`slope`) e talassemia (`thal`).
- Sexo isoladamente não é um bom discriminador, mas pode influenciar em combinação com outros fatores.
- Não foram observados sinais claros de overfitting; as métricas de teste e validação cruzada foram próximas.
- Principais aprendizados: importância do pré-processamento, avaliação de múltiplas métricas, ajuste de hiperparâmetros e trabalho em equipe.

---

## Considerações Sobre Dados Faltantes

- Menos de 5% de dados faltantes em variáveis pouco relevantes: linhas excluídas.
- Dados faltantes em variáveis importantes: imputação com moda.
- Variáveis com muitos valores faltantes (>40%): avaliar se devem ser removidas.
- Se os dados faltantes não forem aleatórios, investigar a causa antes de imputar.

---


## Como Executar

1. Instale as dependências do projeto (veja requirements.txt).
2. Execute o notebook `heartdisease..ipynb` no Jupyter Notebook ou VS Code.
3. Siga as células para análise, modelagem e avaliação dos resultados.

---

## Créditos

Projeto desenvolvido por Rônero Gomes Cavalcante Júnior e Jean Felipe Duarte Tenório
