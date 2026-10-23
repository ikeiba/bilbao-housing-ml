# 🏡 Machine Learning en el Mercado Inmobiliario de Bilbao (`bilbao-housing-ml`)

[![English Version](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-008080)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org/)

> **Cambiar idioma:** **[🇪🇸 Versión en Español](README.es.md)** | [🇬🇧 Read in English](README.md)

---

## 📌 Resumen Ejecutivo

**bilbao-housing-ml** es un proyecto integral de investigación y Machine Learning enfocado en analizar, modelar y segmentar el mercado residencial inmobiliario de **Bilbao (Bizkaia, España)**.

Partiendo del web scraping automatizado de portales inmobiliarios líderes (Idealista), el proyecto aborda los desafíos habituales de los datos del mundo real: valores faltantes (sparsity), desbalanceo de clases, dinámicas de tasación no lineales y texto libre desestructurado. Mediante técnicas avanzadas de imputación con modelos de aprendizaje automático, optimización bayesiana de hiperparámetros (Optuna) y técnicas de explicabilidad (XAI mediante SHAP y ALE), el repositorio abarca tres grandes áreas:
1. **Predicción Precisa de Precios (Regresión):** Estimación del precio de oferta mediante ensamblados de boosting basados en árboles (CatBoost), alcanzando un **MAPE de ~15.0%** y un **$R^2 \approx 0.86$**.
2. **Inferencia de Distritos Urbanos (Clasificación):** Clasificación multiclase de las viviendas entre las 11 zonas de Bilbao utilizando LightGBM e ingeniería de características espaciales y de precio, logrando un **70% de exactitud (accuracy)**.
3. **Segmentación de Tipologías de Vivienda (Clustering):** Descubrimiento de arquetipos de vivienda mediante un pipeline híbrido de densidad **DBSCAN + KNN**, identificando 5 grupos principales y un segmento exclusivo de viviendas de lujo.

---

## 📁 Estructura del Repositorio

```text
bilbao-housing-ml/
│
├── classification/
│   ├── classification_zona.ipynb      # Clasificación multiclase de zonas (LightGBM, Optuna, SHAP, ALE)
│   └── classification_estado.ipynb    # [En curso] Clasificación del estado de conservación (buen estado vs a reformar)
│
├── cleaning/
│   ├── catboost_info/                 # Checkpoints del entrenamiento de modelos de imputación
│   └── data_cleaning.ipynb            # Preprocesamiento, ingeniería de features e imputación basada en ML
│
├── clustering/
│   └── clustering.ipynb               # Segmentación no supervisada del mercado (KMeans, GMM, BIRCH, DBSCAN + KNN)
│
├── data/
│   ├── raw/
│   │   ├── idealista_urls.csv         # URLs de viviendas extraídas en los 11 distritos (1.489 registros)
│   │   └── idealista_viviendas_detalle.csv # Atributos crudos extraídos de las viviendas (1.490 filas x 25 columnas)
│   └── cleaned/
│       └── data_final.csv             # Conjunto de datos final procesado e imputado (1.229 filas x 21 columnas)
│
├── deep_learning/
│   └── deep_learning.ipynb            # [Roadmap] NLP y embeddings a partir de descripciones textuales
│
├── regression/
│   ├── catboost_info/                 # Artefactos y registros del modelo CatBoost
│   └── regression.ipynb               # Pipeline de regresión de precios (EDA, transformaciones log, Optuna, CatBoost, SHAP/ALE)
│
├── scrapers/
│   └── house_info_scraper.ipynb       # Scraper en dos fases con undetected-chromedriver y BeautifulSoup
│
├── README.md                          # Documentación en inglés (English version)
└── README.es.md                       # Documentación en castellano (Versión en español)
```

---

## 🕷️ 1. Adquisición de Datos (Web Scraping)

Los datos brutos fueron recopilados directamente desde **Idealista** mediante un scraper en dos fases implementado en [`scrapers/house_info_scraper.ipynb`](scrapers/house_info_scraper.ipynb):

1. **Fase de Descubrimiento de URLs:**
   - Rastrea los 11 distritos oficiales de Bilbao: *Deusto, Uribarri, Otxarkoaga-Txurdinaga, Basurto-Zorroza, Abando-Albia, Indautxu, Casco Viejo, Begoña-Santutxu, Rekalde, Ibaiondo y San Adrián - La Peña*.
   - Utiliza `undetected_chromedriver` y `selenium` para paginar los resultados de búsqueda esquivando bloqueos anti-bot, generando **1.489 URLs únicas** guardadas en [`data/raw/idealista_urls.csv`](data/raw/idealista_urls.csv).
2. **Fase de Extracción Detallada:**
   - Visita la página de cada inmueble y procesa el HTML con `BeautifulSoup`.
   - Extrae 25 variables iniciales: precio, zona, barrio, metros construidos ($m^2$), metros útiles ($m^2$), habitaciones, baños, planta/piso, exterior/interior, ascensor, garaje, trastero, balcón, obra nueva o segunda mano, estado de conservación, año de construcción, tipo de calefacción, anunciante profesional o particular, valores e índices de consumo y emisiones energéticas (escalas A-G) y descripción textual completa.
   - Guarda los resultados de manera incremental en [`data/raw/idealista_viviendas_detalle.csv`](data/raw/idealista_viviendas_detalle.csv) con manejo de excepciones y tolerancia a caídas.

---

## 🧹 2. Limpieza de Datos e Imputación Avanzada con ML

Los datos inmobiliarios procedentes de portales web presentan valores nulos e inconsistencias. En lugar de descartar filas o aplicar medias globales simples, el equipo implementó estrategias de imputación específicas y basadas en modelos predictivos en [`cleaning/data_cleaning.ipynb`](cleaning/data_cleaning.ipynb):

* **Preprocesamiento General y Deduplicación:** Eliminación de duplicados, estandarización de nombres de columnas a snake_case en inglés, parseo de importes numéricos y tipado estricto.
* **Superficie Útil (`usable_area`):** Comparación entre el ratio promedio por zona frente a modelos de **Regresión Lineal por Zona** para inferir los metros útiles a partir de los metros construidos ($m^2$), garantizando coherencia física ($útil \le construido$).
* **Estado (`condition`):** Identificación de que los valores nulos correspondían a promociones de obra nueva; se imputaron sistemáticamente como `"Nuevo"`.
* **Año de Construcción (`year`):** 479 viviendas de segunda mano no indicaban el año de construcción. Se evaluó un conjunto de modelos:
  - Regresión Lasso Global vs. Lasso + RANSAC (robusto ante valores atípicos).
  - Modelos Lasso y RANSAC segmentados por zona.
  - Modelos basados en árboles: **CatBoostRegressor** y **RandomForestRegressor**.
  - Se seleccionó el mejor modelo por zona (y global con CatBoost/RandomForest para zonas con pocas muestras), preservando la distribución histórica real de cada distrito.
* **Orientación Exterior (`exterior`):** Extracción de términos clave (`exterior`, `terraza`, `balcon`) de la descripción textual. Para los nulos restantes, se entrenó un clasificador **SVM** con ajuste de umbral de probabilidad ($\tau \approx 0.2179$) y manejo de clases desbalanceadas.
* **Planta (`floor`):** Conversión de descripciones cualitativas en una escala numérica ordinal: `-2` (Sótanos / Semisótanos), `-1` (Chalets / Casas unifamiliares), `0` (Plantas bajas / Bajos / Entreplantas), y `1, 2, ...` (plantas estándar).
* **Calificaciones Energéticas (`emissions_label`, `consumption_label`):** Ante la alta ausencia de certificados energéticos oficiales, se formularon como problemas de clasificación multiclase. Se evaluaron Random Forest, Logistic Regression, Naive Bayes y SVM mediante Validación Cruzada Estratificada en 5 folds. El **Random Forest Classifier** obtuvo el mejor rendimiento y estabilidad para imputar las letras energéticas ($A$ hasta $G$).
* **Filtrado de Columnas:** Eliminación de variables con excesiva dispersión o redundancia (`heating`, valores numéricos continuos de emisiones y consumo).
* **Dataset Limpio Final:** **1.229 inmuebles** y **21 atributos estructurados** exportados a [`data/cleaned/data_final.csv`](data/cleaned/data_final.csv).

---

## 📈 3. Pipeline de Regresión de Precios

Desarrollado en [`regression/regression.ipynb`](regression/regression.ipynb) con el objetivo de predecir el precio de mercado de las viviendas con el menor error posible.

### Análisis Exploratorio (EDA) y Transformaciones
- La variable objetivo `price` presentaba una fuerte asimetría positiva (right-skewed) y heterocedasticidad. La aplicación de la transformación logarítmica ($\log(\text{price})$) normalizó la distribución y estabilizó los residuos.
- Se examinaron puntos de apalancamiento (leverage) y correlaciones entre `built_area`, `usable_area` y número de estancias.

### Modelado y Optimización de Hiperparámetros
Se compararon diversas familias de modelos mediante validación cruzada:
1. **Regresión Lineal Simple (MCO):** Comparativa sin transformaciones frente a modelo con transformación logarítmica ($R^2 \approx 0.858$, $\text{MAPE} \approx 17.2\%$).
2. **Modelos Lineales Regularizados:** Ridge, Lasso y ElasticNet ajustados mediante Optuna ($\text{MAPE} \approx 17.01\%$, $R^2 \approx 0.856$).
3. **Modelos No Lineales de Ensamblado:** Random Forest, XGBoost, LightGBM y CatBoost.

### Modelo Campeón: CatBoost Regressor
**CatBoost** superó al resto de algoritmos. Sus hiperparámetros se optimizaron con **Optuna** guiados por la minimización del MAPE en validación:
- **Mejores Hiperparámetros:** `iterations`: 2758, `depth`: 6, `learning_rate`: 0.044, `l2_leaf_reg`: 10.29, `bagging_temperature`: 0.659, `random_strength`: 9.49, `border_count`: 164.

| Métrica | Validación Cruzada / Validación | Conjunto de Test | Evaluación |
| :--- | :---: | :---: | :--- |
| **MAPE (%)** | **15.08%** | **15.70%** | Precisión notable para datos de mercado inmobiliario |
| **$R^2$ Score** | **0.8646** | **0.8005** | Alta capacidad explicativa sin sobreajuste severo |
| **MAE (€)** | **~68.160 €** | **~72.400 €** | Generalización consistente en distintos rangos de precio |

### Explicabilidad e Interpretabilidad (XAI)
- **Importancia de Variables (Feature Importance):** Los cuartos de baño, la superficie útil y la superficie construida son los principales motores de la tasación; el año de construcción y la planta aportan ajustes secundarios.
- **Accumulated Local Effects (ALE):** Confirmación de incrementos monótonos no lineales en el precio a medida que aumentan los metros cuadrados y la altura del piso.
- **SHAP (SHapley Additive exPlanations):**
  - **Gráficos Waterfall Locales:** Explicación individualizada para viviendas concretas, detallando en euros el impacto positivo o negativo de cada característica.
  - **Gráficos Summary / Beeswarm Globales:** Visualización completa de la magnitud y dirección con la que cada atributo empuja la predicción del precio.

---

## 🗺️ 4. Clasificación Multiclase de Zonas

Desarrollada en [`classification/classification_zona.ipynb`](classification/classification_zona.ipynb) para predecir a cuál de las **11 zonas de Bilbao** pertenece una vivienda basándose exclusivamente en sus características intrínsecas (sin utilizar el barrio ni coordenadas GPS).

### Metodología y Estrategias Experimentales
- **Desbalanceo de Clases:** Variación notable de muestras entre zonas céntricas (189 en *Abando-Albia*) y distritos periféricos (28 en *San Adrián - La Peña*, *Otxarkoaga*). Se evaluó mediante Validación Cruzada Estratificada (5-folds) siguiendo métricas ponderadas y macro.
- **Estrategias probadas:**
  - *Estrategia 1:* One-Hot Encoding de variables categóricas en modelos tradicionales.
  - *Estrategia 1.1:* Manejo nativo de categóricas en XGBoost, LightGBM y CatBoost.
  - *Estrategia 1.2:* Reducción de dimensionalidad con Support Vector Machines (kernels lineal y RBF).
  - *Estrategia 1.3:* Balanceo de pesos de clase (`class_weight='balanced'`).
  - *Estrategia 1.4 (Ingeniería de Características):* Creación del ratio `price_m2 = price / built_area`, resultando determinante para discriminar distritos de alta cotización frente a zonas más accesibles.
  - *Estrategia 1.5:* Optimización Bayesiana de Hiperparámetros con **Optuna** (50 trials) ajustando profundidad, número de hojas, learning rate y regularizaciones en LightGBM.

### Modelo Campeón: LightGBM Optimizado
- **Exactitud en Test (Accuracy):** **70.0%**
- **Weighted F1-Score:** **0.70** (Precisión: 0.71, Recall: 0.70)
- **Macro F1-Score:** **0.69** (demostrando solidez incluso en clases minoritarias como Otxarkoaga-Txurdinaga con F1 = 0.86)

```text
                       precision    recall  f1-score   support
         abando-albia       0.75      0.79      0.77        48
      basurto-zorroza       0.70      0.64      0.67        25
      begona-santutxu       0.58      0.73      0.65        15
          casco-viejo       0.88      0.58      0.70        12
               deusto       0.58      0.83      0.68        18
             ibaiondo       0.68      0.68      0.68        31
             indautxu       0.86      0.70      0.78        27
otxarkoaga-txurdinaga       0.86      0.86      0.86         7
              rekalde       0.67      0.61      0.63        33
   san-adrian-la-pena       0.50      0.43      0.46         7
             uribarri       0.70      0.70      0.70        23

             accuracy                           0.70       246
            macro avg       0.70      0.69      0.69       246
         weighted avg       0.71      0.70      0.70       246
```

- **Interpretabilidad:** Análisis mediante gráficos ALE sobre las variables numéricas más influyentes (`year`, `price_m2`, `price`, `usable_area`) e importancia global con valores SHAP.

---

## 🧩 5. Segmentación No Supervisada del Mercado (Clustering)

Desarrollada en [`clustering/clustering.ipynb`](clustering/clustering.ipynb) para encontrar de forma no supervisada las tipologías naturales de vivienda que conviven en Bilbao.

### Evaluación de Algoritmos
- Selección de variables numéricas normalizadas: `price`, `built_area`, `usable_area`, `bedrooms`, `bathrooms`, `floor` y `year`.
- Se evaluaron **K-Means** (método del codo e índice de silueta 0.36 – 0.57), **Gaussian Mixture Models (GMM)** guiados por los criterios BIC y AIC, y **BIRCH**.
- Se seleccionó **DBSCAN** por su capacidad para descubrir agrupaciones de densidad arbitraria sin imponer simetría esférica.

### Tratamiento de Ruido y Outliers mediante KNN
- DBSCAN identificó agrupaciones densas pero dejó 579 instancias en zonas de frontera o baja densidad consideradas ruido.
- Se implementó un algoritmo basado en **K-Nearest Neighbors (KNN)** para analizar las distancias al vecindario más cercano.
- Se fijó un umbral de corte de distancia ($\text{distancia} \le 5.50$):
  - Los puntos frontera fueron reasignados suavemente a su clúster más afín.
  - Los 29 valores extremos reales quedaron aislados en un clúster explícito de **Viviendas de Lujo / Grandes Propiedades (`-1`)**.

### Arquetipos de Vivienda Descubiertos

| Clúster | Cantidad | Precio Medio | Superficie Construida | Hab. / Baños | Año Medio | Perfil y Distribución |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`-1`** | 29 | 1.451.102 € | 377,0 $m^2$ | 7,7 / 3,4 | 1938 | **Mansiones y Áticos Exclusivos:** Propiedades históricas singulares concentradas en Abando-Albia (51,7%) e Indautxu (20,7%). |
| **`0`** | 110 | 273.375 € | 56,5 $m^2$ | 1,0 / 1,0 | 1960 | **Estudios y Apartamentos Urbanos (1 hab):** Viviendas compactas con un 100% de ascensor, ideales para jóvenes profesionales o personas solas. |
| **`1`** | 92 | 415.208 € | 88,8 $m^2$ | 2,0 / 2,0 | 1997 | **Viviendas Modernas / Obra Reciente:** 47,8% obra nueva, 46,7% calificación energética A, 44,6% con garaje y 2 baños completos. |
| **`2`** | 615 | 611.054 € | 140,1 $m^2$ | 3,7 / 2,3 | 1965 | **Pisos Familiares Amplios en Zonas Céntricas:** El segmento con mayor volumen (615 viviendas); pisos señoriales en Abando e Indautxu con balcón (63,4%). |
| **`3`** | 192 | 269.890 € | 85,1 $m^2$ | 3,2 / 1,0 | 1952 | **Pisos Familiares Económicos (3 hab):** Inmuebles tradicionales en distritos obreros/periféricos (Ibaiondo, Santutxu, Rekalde); rara presencia de garaje (6,2%). |
| **`4`** | 191 | 252.653 € | 68,7 $m^2$ | 2,0 / 1,0 | 1948 | **Pisos Compactos Asequibles (2 hab):** Rango de precio más económico en barrios consolidados (Ibaiondo, Basurto, Rekalde); fincas con mayor antigüedad. |

---

## 🔮 6. Trabajo en Curso y Próximos Pasos

- [ ] **`classification/classification_estado.ipynb`:** Desarrollo de modelos de clasificación binaria para discernir si una propiedad está en *buen estado* o requiere reforma integral (*a reformar*) a partir de la antigüedad, precio unitario y planta.
- [ ] **`deep_learning/deep_learning.ipynb`:** Pipeline de Procesamiento del Lenguaje Natural (NLP) sobre las descripciones textuales (`description`):
  - Extracción de términos de alto valor añadido (ej. "vistas despejadas", "luminoso", "reformado de diseño").
  - Extracción de embeddings contextuales (vía transformadores multilingües RoBERTa/BERT) combinados con modelos tabulares para valoración multimodal.

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Entorno de Ejecución:** Python 3.10+
- **Tratamiento de Datos y Álgebra:** `pandas`, `numpy`, `scipy`
- **Visualización:** `matplotlib`, `seaborn`
- **Machine Learning y Preprocesamiento:** `scikit-learn`, `imbalanced-learn`
- **Frameworks de Gradient Boosting:** `catboost`, `lightgbm`, `xgboost`
- **Optimización de Hiperparámetros:** `optuna`
- **Inteligencia Artificial Explicable (XAI):** `shap`, `PyALE`
- **Web Scraping:** `selenium`, `undetected-chromedriver`, `beautifulsoup4`

---

## 👥 Equipo y Contribuciones

Proyecto desarrollado para la asignatura de Machine Learning en la **Universidad de Deusto** (Curso 2024–2025):

- **David Rodríguez** ([@Davidrs05](https://github.com/Davidrs05)) — Pipeline de Regresión de Precios, Imputación del Año de Construcción, Optimización con Optuna y Análisis SHAP/ALE.
- **Enetz Quintana** ([@EnetzUni](https://github.com/EnetzUni)) — Análisis de Imputación de Metros Útiles, Diseño del Roadmap de Deep Learning / NLP.
- **Gonzalo Iglesias** ([@gonzaloiglesias](https://github.com/gonzaloiglesias)) — Pipeline de Clustering del Mercado Inmobiliario (DBSCAN + Reasignación KNN), Imputación de Exterior/Interior.
- **Iker Ibarrola** ([@ikeiba](https://github.com/ikeiba)) — Arquitectura del Scraper, Clasificación Multiclase de Zonas (LightGBM, Optuna, ALE/SHAP), Clasificación de Etiquetas Energéticas.
- **Xabier Villa** ([@XabiVii](https://github.com/XabiVii)) — Procesamiento y Parsing de Plantas, Imputación de Emisiones/Consumo, Roadmap de Clasificación del Estado de Conservación.

---

## 📄 Nota Académica

El contenido de este repositorio ha sido desarrollado con fines exclusivamente académicos y de investigación en el marco del grado universitario en la Universidad de Deusto. Los datos recopilados de fuentes públicas se emplean únicamente para tareas de análisis, modelado estadístico y experimentación educativa.

