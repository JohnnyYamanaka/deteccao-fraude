# Detecção-fraude
Por Johnny Yamanaka
:raising_hand: [LindedIn](https://www.linkedin.com/in/johnny-yamanaka/) | :mailbox_with_no_mail: E-mail: yamanaka.johnny@outlook.com;  

## Sumário
* [Introdução](#introdução);
* [Etapas do Projeto](#etapas-do-projeto)
* [Os Dados](#os-dados)
* [Métricas de Avaliação](#métricas-de-avaliação)
* [Baseline](#baseline)
* [Feature Engineering](#feature-engineering)
* [Tunagem e Entrega do Modelo](#tunagem-e-entrega-do-modelo)
* [Prototipação](#prototipação)

## Introdução
Este projeto tem como objetivo desenvolver um modelo para detectar fraudes em transações financeiras, de forma a causar o menor prejuízo possível.  
 Para isso, será investigado um dataset para encontrar padrões que indiquem comportamentos fraudes.

## Etapas do Projeto
* [Aquisição dos dados](get_data.py);
* [Análise exploratória](notebooks/eda.ipynb);
* [Criação do primeiro modelo e baseline](baseline.py);
* [Feature engineering](features_investigation.py);
* [Tunagem e entrega do modelo](train.py).
* [Prototipação](fraud_detection.py)


## Os Dados
Os dados que serão utilizados para o desenvolvimento do modelo foram geradas sinteticamente, através da PaySim mobile money simulator. O [dataset original](https://www.kaggle.com/datasets/ealaxi/paysim1) pode ser encontrado no Kaggle.  
O dataset completo possui mais de 6 milhões de transações, por conta da restrição de tamanho de arquivo do github foi escohida uma amostragem com 1,5 milhões de registros.

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

## Feature Engineering
Features criadas para tentar melhorar o modelo:
* Agrupar a hora da transação em grupos de três horas;
* Flag para identificar quando o valor das colunas passarem do terceiro quartil:
  * `amount`;
  * `oldbalanceOrg`.
* Flag para identificar se o montante da operação é igual a o saldo inicial da conta origem;
* Variável com o amount ao quadrado;
* Variável com a diferença entre a média do montante e o montante.

Após essa etapa, realizado [treinamento](fe_train.py) com três modelos os resultados foram:
| **#** | **modelo**          | **AUC** | **Total of Loss ($)** |
|------:|---------------------|---------|-----------------------|
|     1 | XGBoost             | 0,917   | 19,488,045.29         |
|     2 | Random Forest       | 0,881   | 23,424,315.31         |
|     3 | Logistic Regression | 0,737   | 66,403,649.59         |

Em seguida vamos verificar o desempenho após realzar as tunagens dos modelos.

## Tunagem e Entrega do Modelo
Para descobrir os melhores hiperparâmetros para cada um dos modelos foi utilizado o método [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), que procura a melhor combinação de forma randomizada (ao contrário do [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), que testa todas as combinações possíveis e demanda um grande processamento).  
O resultado do treinamento após a tunagem foi bem melhor que a etapa anterior:
| **#** | **modelo**          | **AUC** | **Total of Loss ($)** |
|------:|---------------------|---------|-----------------------|
|     1 | XGBoost             | 0,990   | 4,433,541.79          |
|     2 | Random Forest       | 0,990   | 4,433,541.79          |
|     3 | Logistic Regression | 0,990   | 4,433,541.79          |

Agora todos os modelos alcançou quase a nota máxima e diminuiu considerávelmente o potencial prejuízo financeiro. Como todos os modelos alcançaram o mesmo resultado, pode se concluir que dado o conjunto esse foi o máximo que pode ser alcançado.  
 **Então qual modelo escolher?**  
 Já que o resultado foram iguais, em tese poderíamos escolher qualquer modelo. Mas para manter a simplicidade, foi escolhido o modelo que roda o Random Forest.

 ## Prototipação
 Para acessar o modelo rodando e avaliando os outros dias da simulação, basta acessar  esse [link](https://share.streamlit.io/johnnyyamanaka/deteccao-fraude/main/fraud_detection.py)