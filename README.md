# MachineLearningOnAWS

# 1. Pipeline of Data Science and Workflow
>![WorkFlow.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/WorkFlow.drawio.png)

# 2. Data Preparation
# 2-1. Amazon S3 (Ingest Data)
Business success is closely tied to the ability of companies to quickly derive value from their data. Therefore, they are moving to highly scalable, available, secure, and flexible data stores, which they call data lakes.<br>
ビジネスの成功は企業がデータから素早く価値を引き出せるかどうかと密接に関係している。そのため、拡張性、可用性、安全性、柔軟性に優れたデータストアに移行しており、これをデータレイクと呼んでいる。
><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/data-diagram.png" width="720">

# 2-2. Amazon Athena, Amazon Redshift, Amazon EMR (Data Analytics, Transformation and Validation)
Before starting machine learning modeling, perform ad hoc exploration and prototyping to understand the data schema and data quality for solving the problem you are facing.<br>
機械学習のモデリングを始める前に、直面している課題解決に対するデータスキーマとデータ品質を理解するために、アドホックな探索やプロトタイピングを行う。
><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/SparkAndDeequ.png" width="640">

Data, especially datasets spanning multiple years, is never perfect. Data quality tends to degrade over time as applications are updated or retired. Data quality is not necessarily a priority for upstream application teams, so downstream data engineering teams must deal with bad or missing data.<br>
データは決して完璧ではなく、数年にも及ぶようなデータセットはなおさらである。アプリケーションの新機能や改廃に伴い、データの品質は時間とともに低下していく傾向にある。上流のアプリケーションチームにとってデータ品質は必ずしも優先事項ではないため、下流のデータエンジニアリングチームが不良データや欠損データを処理する必要がある。
>![Processing-1.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/Processing-1.png)

Amazon SageMaker processing jobs can run Python scripts on container images using familiar open sources such as Pandas, Scikit-learn, Apache Spark, TensorFlow and XGboost.<br>
そこで、Amazon SageMaker Processing jobは、Pandas, Scikit-learnやApache Spark、TensorFlow、XGboostなどの使い慣れたオープンソースを使って、Pythonスクリプトをコンテナイメージ上で実行することができる。<br>

**- AWS Deequ**<br>
Deequ is a library for analyzing data quality and detecting anomalies using Apache Spark.<br>
Deequは、Apache Sparkを使ってデータの品質を分析し、異常を検知するためのライブラリ。

# 2-2-1. Example using Pandas in Apache Spark
What is [Apach Spark](https://github.com/developer-onizuka/HiveMetastore?tab=readme-ov-file#metastore-in-apache-spark) ?
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
df.fillna(df.mean())
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
- Write the DataFrame to [Parquet](https://github.com/developer-onizuka/HiveMetastore?tab=readme-ov-file#5-1-parquet) File
```
df.write.parquet("s3://bucket/path/to/output_data.parquet")
```

# 2-2-2. SageMaker Compute Instance type
> https://aws.amazon.com/jp/ec2/instance-types/

| Instance Familiy | Features |
| --- | --- |
| **T instance type** | An instance of general-purpose burstable performance<br>  汎用のバースト可能なパフォーマンスのインスタンス。<br> |
| **M instance type** | A general-purpose instance with a good balance of compute, memory, and network bandwidth<br>  計算、メモリ、ネットワーク帯域幅のバランスがよい汎用的なインスタンス。<br> |
| **C instance type** | Instances for compute-balanced workloads that require high-performance CPUs<br>  高性能なCPUが必要な計算制約ワークロード向けインスタンス。<br> |
| **R instance type** | An instance for deploying large datasets in memory, such as Apache Spark<br>  Apache Sparkなどの大規模なデータセットをメモリに展開するためのインスタンス。<br> |
| **P instance type** | A high-performance computing instance that uses less GPU and supports [GPUDirect](https://github.com/developer-onizuka/what_is_GPUDirect-Storage)<br>  GPUを兼ね備える高性能計算インスタンス。<br> |
| **G instance type** | Instances ideal for small, cost-sensitive learning and inference workloads<br>  コスト重視の小規模な学習や推論ワークロードに最適なインスタンス。<br> |


# 2-3. AWS Glue DataBrew (a Low code and Visual Data Preparation Tool)
[AWS Glue DataBrew](https://docs.aws.amazon.com/databrew/latest/dg/what-is.html) is a visual data preparation tool that makes it easier for data analysts and data scientists to clean and normalize data to prepare it for analytics and machine learning (ML). <br>
組み込み変換機能でデータセットの以上を検出し、異常データを変換(無効な値や欠損値を修正)。列の相関性を可視化。<br>
><img src="https://docs.aws.amazon.com/images/databrew/latest/dg/images/databrew-overview-diagram.png" width="720">

Column statistics – On this tab, you can find detailed statistics about each column in your dataset, as shown following.
>![dataset-column-stats.png](https://docs.aws.amazon.com/images/databrew/latest/dg/images/dataset-column-stats.png)

Imputing - fills in missing values for categorical attributes by identifying data patterns in the input dataset. It helps reduce the data quality issues due to incomplete / non-available data.<br>
>![imputing.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/imputing.png)

# 2-4. SageMaker Processing Job (Feature Engineering and Create Training Data)
Run the containerized Scikit-learn execution environment with SageMaker Processing Job and convert text to BERT embedding (vectorization).<br>
コンテナ化されたScikit-learnの実行環境をSageMaker Processing Jobで実行し、テキストをBERT埋め込み(ベクトル化)に変換。<br>

An attempt to interpret which words have similar or distant meanings by vectorizing words.<br>
単語をベクトル化することでどの単語同士が近い意味を持つのか遠い意味を持つのかを解釈しようとする試み。<br>
>https://www.youtube.com/watch?v=FoY1rRH2Jc4

><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/Queen.png" width="320">

>https://github.com/oreilly-japan/data-science-on-aws-jp <br>
><img src="https://raw.githubusercontent.com/oreilly-japan/data-science-on-aws-jp/7c1ea12f23725d5dfcc2db989a62bccbcd044340/workshop/00_quickstart/img/prepare_dataset_bert.png" width="720">

> https://sagemaker.readthedocs.io/en/v2.5.5/amazon_sagemaker_processing.html<br>

(1) You can run a scikit-learn script to do data processing on SageMaker using the sagemaker.sklearn.processing.SKLearnProcessor class.
```
from sagemaker.sklearn.processing import SKLearnProcessor
sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role=role,
                                     instance_count=1,
                                     instance_type='ml.m5.xlarge')
```
(2) Then you can run a scikit-learn script preprocessing.py in a processing job. In this example, our script takes one input from S3 and one command-line argument, processes the data, then splits the data into two datasets for output. When the job is finished, we can retrive the output from S3.
- データセット（dataset.csv） は自動的にコンテナの中の所定のディレクトリ （/input）にコピーされる。
- 前処理を行い、データセットを3つに分割し、それぞれのファイルはコンテナの中の /opt/ml/processing/output/train 、 /opt/ml/processing/output/validation 、 /opt/ml/processing/output/test へ保存される。
- ジョブが完了するとその出力を、S3 の中にある SageMaker のデフォルトのバケットへ自動的にコピーされる。
```
from sagemaker.processing import ProcessingInput, ProcessingOutput
sklearn_processor.run(
    code='preprocessing.py',
    # arguments = ['arg1', 'arg2'],
    inputs=[ProcessingInput(
        source='dataset.csv',
        destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(source='/opt/ml/processing/output/train'),
        ProcessingOutput(source='/opt/ml/processing/output/validation'),
        ProcessingOutput(source='/opt/ml/processing/output/test')]
)
```

File locations of containers running in SageMaker are mapped to S3.<br>
SageMakerで実行されるコンテナのファイルロケーションはS3にマッピングされる。
>![SageMakerFileLocation.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/SageMakerFileLocation.drawio.png)


# 3. Model Train and Tuning
# 3-1. BERTなどのNLPモデル

>![Word2vec.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/Word2vec.drawio.png)

# 3-2. SageMaker JumpStart (Fine Tuning)
Fine-tune using the BERT embedding from the review_body text already generated in 2-4 and create a custom classifier that predicts star_rating as shown in the figure below.<br>
既に2-4で生成済みのreview_bodyテキストからのBERT埋め込みを使用してファインチューニングし、以下図に示すようにstar_ratingを予測するカスタム分類器を作成する。
>![StarRating.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/StarRating.drawio.png)


# 4. Deploy and Monitoring


