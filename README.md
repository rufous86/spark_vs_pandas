# Pyspark. Анализ больших данных, когда Pandas не достаточно

  Pandas is one of the most used open-source Python libraries to work with Structured tabular data for analysis. Pandas library is heavily used for Data Analytics, Machine learning, data science projects, and many more.  

  Pandas can load the data by reading CSV, JSON, SQL, many other formats and creates a DataFrame which is a structured object containing rows and columns (similar to SQL table).

  It doesn’t support distributed processing hence you would always need to increase the resources when you need additional horsepower to support your growing data.Pandas DataFrame’s are mutable and are not lazy, statistical functions are applied on each column by default.  

  In very simple words Pandas run operations on a single machine whereas PySpark runs on multiple machines. If you are working on a Machine Learning application where you are dealing with larger datasets, PySpark is a best fit which could processes operations many times(100x) faster than Pandas.  

  PySpark is very well used in Data Science and Machine Learning community as there are many widely used data science libraries written in Python including NumPy, TensorFlow also used due to their efficient processing of large datasets. PySpark has been used by many organizations like Walmart, Trivago, Sanofi, Runtastic, and many more.

  PySpark is a Spark library written in Python to run Python applications using Apache Spark capabilities. Using PySpark we can run applications parallelly on the distributed cluster (multiple nodes) or even on a single node.

  Apache Spark is an analytical processing engine for large scale powerful distributed data processing and machine learning applications.
