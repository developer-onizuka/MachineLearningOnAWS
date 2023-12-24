# MachineLearningOnAWS

# 1. Pipeline of Data Science and Workflow

![WorkFlow.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/WorkFlow.drawio.png)

# 1-1. Amazon S3 (Ingest Data)
Business success is closely tied to the ability of companies to quickly derive value from their data. Therefore, they are moving to highly scalable, available, secure, and flexible data stores, which they call data lakes.<br>
ビジネスの成功は企業がデータから素早く価値を引き出せるかどうかと密接に関係している。そのため、拡張性、可用性、安全性、柔軟性に優れたデータストアに移行しており、これをデータレイクと呼んでいる。

![data-diagram.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/data-diagram.png)

# 1-2. Amazon Athena, Amazon Redshift, Amazon EMR (Data Transformation and Analytics)
Before starting machine learning modeling, perform ad hoc exploration and prototyping to understand the data schema and data quality for solving the problem you are facing.<br>
機械学習のモデリングを始める前に、直面している課題解決に対するデータスキーマとデータ品質を理解するために、アドホックな探索やプロトタイピングを行う。

![SparkAndDeequ](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/SparkAndDeequ.png)

Data, especially datasets spanning multiple years, is never perfect. Data quality tends to degrade over time as applications are updated or retired. Data quality is not necessarily a priority for upstream application teams, so downstream data engineering teams must deal with bad or missing data.<br>
データは決して完璧ではなく、数年にも及ぶようなデータセットはなおさらである。アプリケーションの新機能や改廃に伴い、データの品質は時間とともに低下していく傾向にある。上流のアプリケーションチームにとってデータ品質は必ずしも優先事項ではないため、下流のデータエンジニアリングチームが不良データや欠損データを処理する必要がある。

![Processing-1.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/Processing-1.png)

Amazon SageMaker processing jobs can run Python scripts on container images using familiar open sources such as Pandas, Scikit-learn, Apache Spark, and XGboost.<br>
そこで、Amazon SageMaker Processing jobは、Pandas, Scikit-learnやApache Spark、XGboostなどの使い慣れたオープンソースを使って、Pythonスクリプトをコンテナイメージ上で実行することができる。<br>

# Example using Pandas in Apache Spark
```
%% cat input_data.csv
1,Apple,10.99
2,Orange,11.99
3,Banana,12.99
4,Lemon,NaN
```
- Load the CSV file into DataFrame in Apache Spark
```
df = spark.read.option('header','false').format("csv").load("s3://bucket/path/to/input_data.csv")
df.show()
```
```
+---+------+-----+
|_c0|   _c1|  _c2|
+---+------+-----+
|  1| Apple|10.99|
|  2|Orange|11.99|
|  3|Banana|12.99|
|  4| Lemon|NaN  |
+---+------+-----+
```
- Remove null values ​​as rows
```
df_dropped = df.dropna()
df_dropped.show()
```
```
+---+------+-----+
|_c0|   _c1|  _c2|
+---+------+-----+
|  1| Apple|10.99|
|  2|Orange|11.99|
|  3|Banana|12.99|
+---+------+-----+
```
- Fill in the blanks with the average value
```
df.fillna(df.mean()))
df.show()
```
```
+---+------+-----+
|_c0|   _c1|  _c2|
+---+------+-----+
|  1| Apple|10.99|
|  2|Orange|11.99|
|  3|Banana|12.99|
|  4| Lemon|11.99| <--
+---+------+-----+
```
- Write the DataFrame to CSV File
```
df.write.csv("s3://bucket/path/to/output_data.csv")
```
- Write the DataFrame to Parquet File
```
df.write.parquet("s3://bucket/path/to/output_data.parquet")
```

# 1-2-1. AWS Glue DataBrew (A Visual Data Preparation Tool)
[AWS Glue DataBrew](https://docs.aws.amazon.com/databrew/latest/dg/what-is.html) is a visual data preparation tool that makes it easier for data analysts and data scientists to clean and normalize data to prepare it for analytics and machine learning (ML). 
![overview.png](https://docs.aws.amazon.com/images/databrew/latest/dg/images/databrew-overview-diagram.png)

# 1-3. SageMaker Compute Instance type
> https://aws.amazon.com/jp/ec2/instance-types/

| Instance Familiy | Features |
| --- | --- |
| **T instance type** | An instance of general-purpose burstable performance<br>  汎用のバースト可能なパフォーマンスのインスタンス。<br> |
| **M instance type** | A general-purpose instance with a good balance of compute, memory, and network bandwidth<br>  計算、メモリ、ネットワーク帯域幅のバランスがよい汎用的なインスタンス。<br> |
| **C instance type** | Instances for compute-balanced workloads that require high-performance CPUs<br>  高性能なCPUが必要な計算制約ワークロード向けインスタンス。<br> |
| **R instance type** | An instance for deploying large datasets in memory, such as Apache Spark<br>  Apache Sparkなどの大規模なデータセットをメモリに展開するためのインスタンス。<br> |
| **P instance type** | A high-performance computing instance that uses less GPU<br>  GPUを兼ね備える高性能計算インスタンス。<br> |
| **G instance type** | Instances ideal for small, cost-sensitive learning and inference workloads<br>  コスト重視の小規模な学習や推論ワークロードに最適なインスタンス。<br> |
