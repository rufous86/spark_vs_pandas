# Pyspark. Анализ больших данных, когда Pandas не достаточно

Pandas - одна из наиболее используемых библиотек Python с открытым исходным кодом для работы со структурированными табличными данными для анализа. Библиотека Pandas активно используется для аналитики данных, машинного обучения, проектов в области науки о данных и многих других.

Однако он не поддерживает распределенную обработку, поэтому вам всегда придется увеличивать ресурсы, когда вам понадобится дополнительная мощность для поддержки растущих данных.

Проще говоря, Pandas выполняет операции на одной машине, в то время как PySpark работает на нескольких машинах. Если вы работаете над приложением машинного обучения, где вы имеете дело с большими наборами данных, PySpark является лучшим вариантом, который может обрабатывать операции во много раз (100x) быстрее, чем Pandas.

PySpark - это библиотека Spark, написанная на языке Python для запуска приложений Python с использованием возможностей Apache Spark. Используя PySpark, мы можем запускать приложения параллельно на распределенном кластере (несколько узлов) или даже на одном узле.

Скачаем необходимые наборы данных с kaggle  
инструкция по скачиванию данных в colab напрямую с kaggle ниже:  
https://www.kaggle.com/general/74235

![](https://github.com/rufous86/spark_vs_pandas/blob/main/assets/kaggle_token.png?raw=1)


```pyton

# ! pip install -q kaggle
# from google.colab import drive
# drive.mount('/content/drive')
# ! mkdir ~/.kaggle
# ! cp '/content/drive/MyDrive/Colab Notebooks/kaggle.json' ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle competitions download -c 'riiid-test-answer-prediction'
# ! mkdir data
# ! unzip riiid-test-answer-prediction.zip -d data

# ! pip install pyspark
# ! pip install pyarrow

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

df = spark.read.csv('data/train.csv', header=True, inferSchema=True)

df.printSchema()

df.show()

from pyspark.sql.types import IntegerType

df = df.withColumn('prior_question_had_explanation', df['prior_question_had_explanation'].cast(IntegerType()))
df.show()
df.printSchema()

df.pandas_api().isna().sum()

"""Посмотрим, сколько в нашей таблице пустых значений"""

df = df.dropna()
df.pandas_api().isna().sum()

"""Проанализируем характеристики, влияющие на успеваемость студентов. Так как фактически данные об успеваемости у нас отсутствуют, условно за успеваемость будут выступать правильно данные ответы.      
Сначала сохраним колонку answered_correctly в переменную target. Это будет наша целевая переменная  
Затем рассчитаем коэффициент корреляции целевой переменной с каждой из остальных характеристик
"""

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

cor_np = matrix.collect()[0][matrix.columns[0]].toArray()

import pandas as pd

corr_matrix_df = pd.DataFrame(data=cor_np, columns = df.columns, index=df.columns)

import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plt.figure(figsize=(16,5))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True)
plt.show()
```
