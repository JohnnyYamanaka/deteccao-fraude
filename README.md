# Detecção-fraude
Por Johnny Yamanaka
:raising_hand: [LindedIn](https://www.linkedin.com/in/johnny-yamanaka/) | :mailbox_with_no_mail: E-mail: yamanaka.johnny@outlook.com;  

## Introdução
Projeto de desenvolvimento de um modelo para detecção de fraudes em transações financeiras.  


## Os Dados
Os dados que serão utilizados para o desenvolvimento do modelo foi gerada sinteticamente, através da PaySim mobile money simulator. O [dataset original](https://www.kaggle.com/datasets/ealaxi/paysim1) pode ser encontrado no Kaggle.  
O dataset completo possui mais de 6 milhões de transações, por conta da restrição de tamanho de arquivo do github foi escohida uma amostragem com 1,5 milhões de registros.

## Etapas do Projeto
* [Aquisição dos dados](get_data.py);
* [Análise exploratória](notebooks/eda.ipynb);
* [Criação do primeiro modelo e baseline](baseline.py);
* Feature engineering;
* Tunagem e entrega do modelo.


## Métricas de Avaliação
Para avaliar o desempenho dos modelos, serão aplicadas duas métricas principais:
* **AUC** (Area Under the Curve) - Mede o quão bem o modelo é capaz de separar as classes: se a transação é uma fraude ou não. O valor varia de 0 a 1 e quanto mais próximo de 1 o resultado for, significa que o modelo é capaz de separar da melhor forma;
* **Total of Loss** - Valor total de prejuízo (valor da transação) caso o modelo erre em classificar uma transação fraudulenta.

## Baseline
Para fim de comparação, foram utilizados cinco modelos para verificar criar uma baseline e o resultado obtido foram, na ordem do melhor para o pior:

| **#** | **modelo**          | **AUC** | **Total of Loss ($)** |
|------:|---------------------|---------|-----------------------|
|     1 | XGBoost             | 0,915   | 20.631.068,79         |
|     2 | Random Forest       | 0,883   | 23.406.744,23         |
|     3 | Logistic Regression | 0,726   | 73.417.490,10         |
|     4 | LightGBM            | 0,699   | 593.543.581,71        |
|     5 | DummyClassifier     | 0,500   | 567.114.634,55        |

Mesmo o melhor modelo ainda representa um valor muito alto de prejuízo ($ 20M).  
A seguir vamos realizar feature engineering no dataset e a tunagem de modelo para verificar se é possível melhorar o desempenho do modelo.
