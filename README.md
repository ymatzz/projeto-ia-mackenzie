# Descrição do Dataset, Conteúdo/Origem, Análise Exploratória e Preparação dos Dados em Python

Para a validação inicial da pipeline de pré-processamento e análise exploratória de dados (EDA) na entrega do primeiro bimestre (N1), este trabalho adota temporariamente como *benchmark* a base de dados pública **UI-PRMD** (*University of Idaho - Physical Rehabilitation Movement Dataset*), especificamente o subconjunto referente ao exercício de **Agachamento Profundo (*Deep Squat*)** [1]. A adoção desta base de referência justifica-se pela necessidade de homologar a arquitetura analítica enquanto a captura autoral dos dados cinemáticos dos membros do grupo está em fase de execução técnica.

## Origem e Estrutura dos Dados

O subconjunto é composto por registros de movimento de 10 indivíduos realizando repetições do exercício de agachamento em duas modalidades técnicas: **execução correta** (padrão biomecânico ideal) e **execução incorreta** (simulação de desvios posturais e compensações físicas comuns). A base disponibilizada é composta por seis arquivos CSV estruturados:

* **`Data_Correct.csv` e `Data_Incorrect.csv`** (Dimensão: \(10.530 \times 240\)):
  * Contêm as matrizes de dados cinemáticos brutos (ângulos articulares tridimensionais) alinhados temporalmente em 240 *frames* por repetição.
  * O conjunto totaliza 90 sequências de movimento (repetições), onde cada repetição contém 117 variáveis/ângulos articulares (\(90 \text{ repetições} \times 117 \text{ ângulos} = 10.530\) linhas).

* **`Autoencoder_Output_Correct.csv` e `Autoencoder_Output_Incorrect.csv`** (Dimensão: \(90 \times 960\)):
  * Representam o espaço latente de dimensão reduzida extraído via rede neural *Autoencoder*.
  * Reduz os 117 ângulos brutos para 4 características latentes essenciais por *frame* (\(4 \text{ características} \times 240 \text{ frames} = 960\) colunas por execução).

* **`Labels_Correct.csv` e `Labels_Incorrect.csv`** (Dimensão: \(90 \times 1\)):
  * Apresentam os escores contínuos de qualidade e conformidade do movimento (*Quality Scores*), variando no intervalo \([0, 1]\), calculados via Modelos de Mistura Gaussiana (*Gaussian Mixture Models* - GMM).
  * Execuções corretas apresentam média de qualidade alta (\(\mu = 0,944; \sigma = 0,011\)), enquanto execuções incorretas exibem maior dispersão e menor qualidade média (\(\mu = 0,860; \sigma = 0,067\)).

---

## Referências
\[1\] University of Idaho - Physical Rehabilitation Movement Dataset (UI-PRMD).
