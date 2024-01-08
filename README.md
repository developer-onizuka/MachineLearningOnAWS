# Goal
**- Understand the difference between Pre-Training and Fine-Tuning** <br>
**- Understand what is Feature Engineering** <br>

>![bert_training.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/bert_training.png)

# 1. Pipeline of Data Science and Workflow
>![WorkFlow.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/WorkFlow.drawio.png)

# 2. Data Preparation
# 2-1. Amazon S3 (Ingest Data into Data Lake)
Business success is closely tied to the ability of companies to quickly derive value from their data. Therefore, they are moving to highly scalable, available, secure, and flexible data stores, which they call data lakes.<br>
ビジネスの成功は企業がデータから素早く価値を引き出せるかどうかと密接に関係している。そのため、拡張性、可用性、安全性、柔軟性に優れたデータストアに移行しており、これをデータレイクと呼んでいる。
><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/aws-data-lake.png" width="720">

# 2-2. Amazon Athena, Amazon Redshift, Amazon EMR (Data Analytics, Transformation and Validation)
Before starting machine learning modeling, perform ad hoc exploration and prototyping to understand the data schema and data quality for solving the problem you are facing.<br>
機械学習のモデリングを始める前に、直面している課題解決に対するデータスキーマとデータ品質を理解するために、アドホックな探索やプロトタイピングを行う。
><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/SparkAndDeequ.png" width="640">

Data, especially datasets spanning multiple years, is never perfect. Data quality tends to degrade over time as applications are updated or retired. Data quality is not necessarily a priority for upstream application teams, so downstream data engineering teams must deal with bad or missing data.<br>
データは決して完璧ではなく、数年にも及ぶようなデータセットはなおさらである。アプリケーションの新機能や改廃に伴い、データの品質は時間とともに低下していく傾向にある。上流のアプリケーションチームにとってデータ品質は必ずしも優先事項ではないため、下流のデータエンジニアリングチームが不良データや欠損データを処理する必要がある。
>![Processing-1.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/Processing-1.png)

Amazon SageMaker processing jobs can run Python scripts on container images using familiar open sources such as Pandas, Scikit-learn, Apache Spark, TensorFlow and XGboost.<br>
そこで、Amazon SageMaker Processing jobは、Pandas、Scikit-learnやApache Spark、TensorFlow、XGboostなどの使い慣れたオープンソースを使って、Pythonスクリプトをコンテナイメージ上で実行することができる。<br>

**- AWS Deequ**<br>
Deequ is a library for analyzing data quality and detecting anomalies using Apache Spark.<br>
Deequは、Apache Sparkを使ってデータの品質を分析し、異常を検知するためのライブラリ。

# 2-2-1. Example using Pandas
See Fruits.ipynb in this repository.

# 2-2-2. Example using Apache Spark with Amazon review Dataset
You can learn about Apach Spark [here](https://github.com/developer-onizuka/HiveMetastore?tab=readme-ov-file#metastore-in-apache-spark).
If you wanna create your own evironment for container, then you can create it so easily on your windows10 machine. See it the link below:<br>
>https://github.com/developer-onizuka/NvidiaDockerOnWSL<br>

# (1) Run a Spark Container
```
$ sudo docker run -it --rm -p 8888:8888 --name spark jupyter/all-spark-notebook:spark-3.5.0
```
If you want to use volume which can be shared between a host and containers and use GPU, try the command below:
```
$ sudo docker run -it -v /mnt/c/Temp:/mnt --rm --gpus all -p 8888:8888 --name spark jupyter/all-spark-notebook:spark-3.5.0
```
You can connect to the Jupyter notebook with your browser.

# (2) Create a Spark Session
```
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

spark = SparkSession \
        .builder \
        .appName("myapp") \
        .master("local") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.parquet.binaryAsString","true") \
        .getOrCreate()

conf = spark.sparkContext.getConf()
print("# spark.executor.memory = ", conf.get("spark.executor.memory"))
print("# spark.executor.memoryOverhead = ", conf.get("spark.executor.memoryOverhead"))
# spark.executor.memory =  8g
# spark.executor.memoryOverhead =  None
```
# (3) Load the Amazon review Dataset into DataFrame of Apache Spark
```
df=spark.read.parquet("/mnt/amazon_reviews_2015.snappy.parquet")
# https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet
```
# (4) Confirm the Data and its Schema
```
df.printSchema()
root
 |-- review_date: integer (nullable = true)
 |-- marketplace: string (nullable = true)
 |-- customer_id: decimal(20,0) (nullable = true)
 |-- review_id: string (nullable = true)
 |-- product_id: string (nullable = true)
 |-- product_parent: decimal(20,0) (nullable = true)
 |-- product_title: string (nullable = true)
 |-- product_category: string (nullable = true)
 |-- star_rating: short (nullable = true)
 |-- helpful_votes: long (nullable = true)
 |-- total_votes: long (nullable = true)
 |-- vine: boolean (nullable = true)
 |-- verified_purchase: boolean (nullable = true)
 |-- review_headline: string (nullable = true)
 |-- review_body: string (nullable = true)
```
```
df.show(10)
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
|review_date|marketplace|customer_id|     review_id|product_id|product_parent|       product_title|product_category|star_rating|helpful_votes|total_votes| vine|verified_purchase|     review_headline|         review_body|
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
|      16455|         US|   47052105|R2C20GSMIOZYVP|B004BQWJXK|     111796163|Prosciutto Di Par...|         Grocery|          5|            2|          2|false|             true|you will not be d...|I have made multi...|
|      16455|         US|   49070565| RPI30SPP1J9U9|B00IJGSSRO|      15606588|Hoosier Hill Farm...|         Grocery|          3|            1|          1|false|             true|Not sure if the q...|I am not sure if ...|
|      16455|         US|    3081462| RKYY2ZQGUV06L|B004QJEJ1M|     179477362|Limited Edition M...|         Grocery|          1|            0|          0|false|             true|Taste is as mild ...|I was  hoping thi...|
|      16455|         US|   13807093| RKYYAEA9G3CD4|B00INK53D8|     600604708|Stash Tea Teabags...|         Grocery|          5|            0|          0|false|             true|          Five Stars|        Awesome Tea!|
|      16455|         US|   48608794|R17ZQPU555KVR6|B00P0LLZ4O|     454046525|Reese's Spreads P...|         Grocery|          4|            0|          0|false|            false|This tasty spread...|This tasty spread...|
|      16455|         US|   45849122|R1Q5A9D8NWQDXZ|B000V9EIEE|     282714980|Nips Candy (pack ...|         Grocery|          5|            0|          0|false|             true|  The best snack yet|        I love these|
|      16455|         US|   45430661| RB4ZEYM0KIH2L|B00BXMHZNO|     843662630|ICE CHIPS Pepperm...|         Grocery|          5|            0|          0|false|             true|          Five Stars|They are wonderfu...|
|      16455|         US|   42665705|R11NTR1JWBJK19|B004N5MDY4|     479154054|HERSHEY'S Holiday...|         Grocery|          3|            0|          0|false|            false|It was good, but ...|It was Hershey's ...|
|      16455|         US|   39221810| RURVG71IJE5TD|B00K89HCTU|     253806662|Coffee Variety Sa...|         Grocery|          5|            0|          0|false|            false|        A Good Value|The 40-count coff...|
|      16455|         US|   16143433|R1B96XLD73K1OS|B000WV0RW8|     653213046|Healthworks Chia ...|         Grocery|          4|            1|          1|false|             true|          Four Stars|             perfect|
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
only showing top 10 rows
```
```
df.registerTempTable('views')
temp = spark.sql('SELECT * FROM views WHERE star_rating=3')
temp.show(10)
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
|review_date|marketplace|customer_id|     review_id|product_id|product_parent|       product_title|product_category|star_rating|helpful_votes|total_votes| vine|verified_purchase|     review_headline|         review_body|
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
|      16455|         US|   49070565| RPI30SPP1J9U9|B00IJGSSRO|      15606588|Hoosier Hill Farm...|         Grocery|          3|            1|          1|false|             true|Not sure if the q...|I am not sure if ...|
|      16455|         US|   42665705|R11NTR1JWBJK19|B004N5MDY4|     479154054|HERSHEY'S Holiday...|         Grocery|          3|            0|          0|false|            false|It was good, but ...|It was Hershey's ...|
|      16455|         US|   18796614|R1RPQ7HKR2FL3S|B0052P27E0|     461603490|Newman's Own Orga...|         Grocery|          3|            0|          0|false|             true|         Three Stars|Good...but the mo...|
|      16455|         US|   44721885|R1EW92YL5PUGHG|B000FED3SW|     973710819|Red Vines Sugar F...|         Grocery|          3|            0|          0|false|             true|         Three Stars|I used to love th...|
|      16455|         US|    8000689|R1NSHT693NNN53|B008F8BNIM|       1228503|King of Joe Cappu...|         Grocery|          3|            0|          0|false|             true|         Three Stars|                Okay|
|      16455|         US|   36920989|R1070HUW6G1PSQ|B0083CP20A|     204990765|Lily's Dark Choco...|         Grocery|          3|            1|          1|false|             true|Wish they'd leave...|Wish they'd  leav...|
|      16455|         US|   49752507|R1BUB8YD8ARX25|B0050IBIJE|     765113109|Kong Company Real...|         Grocery|          3|            0|          0|false|             true|      It works. But.|It's peanut butte...|
|      16455|         US|   37376084|R3QK3FZ3YYQ9XZ|B00JWZS6N2|     198841106|Jelly Belly Draft...|         Grocery|          3|            0|          0|false|             true|         Three Stars|Tastes like beer!...|
|      16455|         US|   22429897|R2S3A8LM8WCNRR|B008JEK8XI|      94923674|Sugar Free Red Wi...|         Grocery|          3|            0|          0|false|             true|         Three Stars|    They are good :)|
|      16455|         US|   42968311| ROD0XDNDI5IBW|B008Z5L2MW|     532116530|Starbucks&#0174; ...|         Grocery|          3|            0|          0|false|             true|         Three Stars|Good but slightly...|
+-----------+-----------+-----------+--------------+----------+--------------+--------------------+----------------+-----------+-------------+-----------+-----+-----------------+--------------------+--------------------+
only showing top 10 rows
```
# (5) Check Null values
In case of some errors, you may increase the spark.driver.memory and spark.executor.memory.
```
from pyspark.sql.functions import *
#count number of null values in each column of DataFrame
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
+-----------+-----------+-----------+---------+----------+--------------+-------------+----------------+-----------+-------------+-----------+----+-----------------+---------------+-----------+
|review_date|marketplace|customer_id|review_id|product_id|product_parent|product_title|product_category|star_rating|helpful_votes|total_votes|vine|verified_purchase|review_headline|review_body|
+-----------+-----------+-----------+---------+----------+--------------+-------------+----------------+-----------+-------------+-----------+----+-----------------+---------------+-----------+
|          0|          0|          0|        0|         0|             0|            0|               0|          0|            0|          0|   0|                0|              0|          0|
+-----------+-----------+-----------+---------+----------+--------------+-------------+----------------+-----------+-------------+-----------+----+-----------------+---------------+-----------+
```

# 2-2-3. SageMaker Compute Instance type
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
組み込み変換機能でデータセットの異常を検出し、異常データを変換(無効な値や欠損値を修正)。列の相関性を可視化。<br>
><img src="https://docs.aws.amazon.com/images/databrew/latest/dg/images/databrew-overview-diagram.png" width="720">

Column statistics – On this tab, you can find detailed statistics about each column in your dataset, as shown following.
>![dataset-column-stats.png](https://docs.aws.amazon.com/images/databrew/latest/dg/images/dataset-column-stats.png)

Imputing - fills in missing values for categorical attributes by identifying data patterns in the input dataset. It helps reduce the data quality issues due to incomplete / non-available data.<br>
>![imputing.png](https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/imputing.png)

# 2-4. SageMaker Processing Job (Feature Engineering and Create Training Data)
Perform feature engineering to create a BERT embedding from the review_body text using the pretrained BERT model and split the dataset into training, validation, and test files.<br>
事前学習済みBERTモデルを使用して review_body テキストからBERT埋め込みを作成するための特徴量エンジニアリングを実行し、データセットをトレーニング、バリデーション、およびテストファイルに分割する。

# 2-4-1. What is vectorization ?
Run the containerized Scikit-learn execution environment with SageMaker Processing Job and convert text to BERT embedding (vectorization).<br>
コンテナ化されたScikit-learnの実行環境をSageMaker Processing Jobで実行し、テキストをBERT埋め込み(ベクトル化)に変換。<br>

An attempt to interpret which words have similar or distant meanings by vectorizing words.<br>
単語をベクトル化することでどの単語同士が近い意味を持つのか遠い意味を持つのかを解釈しようとする試み。<br>
>https://www.youtube.com/watch?v=FoY1rRH2Jc4

><img src="https://github.com/developer-onizuka/MachineLearningOnAWS/blob/main/Queen.png" width="320">

The vector value is a value in a general-purpose linguistic knowledge space obtained through pre-learning (learning with a large corpus using specific words from existing texts such as Wikipedia as input and the preceding and following words as training data). <br>
なお、当該ベクトル値は、事前学習(Wikipediaなどの既存の文章における特定の単語を入力とし、前後の単語を教師データとした大規模なコーパスでの学習)で得られた汎用的な言語知識空間内に定義された値となる。
当該デフォルトのBERTモデルをファインチューニングすることで、Amazon Customer Reviews Datasetなどに対するカスタムテキスト分類器を作成する。言い換えると、事前学習済BERTモデルによって学習された言語理解と意味論を再利用することで、新しい領域固有のNLPタスクを学習することが狙い。<br>
そのために必要になってくる生のテキストをBERT埋め込みに変換するためのスニペットを2-5に示す。<br>

# 2-4-2. Data Pre-Processing with Scikit-learn
>https://github.com/oreilly-japan/data-science-on-aws-jp <br>
><img src="https://raw.githubusercontent.com/oreilly-japan/data-science-on-aws-jp/7c1ea12f23725d5dfcc2db989a62bccbcd044340/workshop/00_quickstart/img/prepare_dataset_bert.png" width="720">

> https://sagemaker.readthedocs.io/en/v2.5.5/amazon_sagemaker_processing.html<br>

# (1) Instance for SageMaker
You can run a scikit-learn script to do data processing on SageMaker using the sagemaker.sklearn.processing.SKLearnProcessor class.
```
from sagemaker.sklearn.processing import SKLearnProcessor

sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role=role,
                                     instance_count=1,
                                     instance_type='ml.m5.xlarge')
```
# (2) Input/Output for SageMaker
Then you can run a scikit-learn script preprocessing.py in a processing job. In this example, our script takes one input from S3 and one command-line argument, processes the data, then splits the data into two datasets for output. When the job is finished, we can retrive the output from S3.
- データセット（dataset.csv） は自動的にコンテナの中の所定のディレクトリ （/input）にコピーされる。
- 前処理を行い、データセットを3つに分割し、それぞれのファイルはコンテナの中の /opt/ml/processing/output/train 、 /opt/ml/processing/output/validation 、 /opt/ml/processing/output/test へ保存される。
- ジョブが完了するとその出力を、S3 の中にある SageMaker のデフォルトのバケットへ自動的にコピーされる。
```
from sagemaker.processing import ProcessingInput, ProcessingOutput
sklearn_processor.run(
    code='preprocess.py',
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

# 2-4-3. Data Pre-Processing with Spark
```
from sagemaker.processing import ScriptProcessor, ProcessingInput

spark_processor = ScriptProcessor(
    base_job_name="spark-preprocessor",
    image_uri="<ECR repository URI to your Spark processing image>",
    command=["/opt/program/submit"],
    role=role,
    instance_count=2,
    instance_type="ml.r5.xlarge",
    max_runtime_in_seconds=1200,
    env={"mode": "python"},
)
```
```
spark_processor.run(
    code="preprocess.py",
    arguments=[
        "s3_input_bucket",
        bucket,
        "s3_input_key_prefix",
        input_prefix,
        "s3_output_bucket",
        bucket,
        "s3_output_key_prefix",
        input_preprocessed_prefix,
    ],
    logs=False,
)
```

# 2-5. Convert raw text to BERT features
Spark snippet to convert raw text to BERT embedding using Transformers provided as a Python library<br>
Pythonライブラリとして提供されているTransformersを使い、生のテキストをBERT埋め込みに変換するSparkのスニペット。2-4-2、2-4-3における"preprocess.py"の中身に相当するもの。<br>
# (1) Define Tokenizer with BERT
```
import tensorflow as tf
import collections
import json
import os
import pandas as pd
import csv
from transformers import DistilBertTokenizer

max_seq_length = 64

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

REVIEW_BODY_COLUMN = "review_body"
REVIEW_ID_COLUMN = "review_id"

LABEL_COLUMN = "star_rating"
LABEL_VALUES = [1, 2, 3, 4, 5]
```
# (2) Tokenize the input thru Tokenizer defined above
```
def convert_input(the_input, max_seq_length):
    # まず、BERTが学習したデータ形式と合うようにデータを前処理する。
    # 1. テキストを小文字にする（BERT lowercaseモデルを用いる場合）
    # 2. トークン化する（例、"sally says hi" -> ["sally", "says", "hi"]）
    # 3. 単語をWordPieceに分割（例、"calling" -> ["call", "##ing"]）
    #
    # この辺りの処理はTransformersライブラリのトークナイザーがまかなってくれます。

    tokens = tokenizer.tokenize(the_input.text)
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    # print("**{} tokens**\n{}\n".format(len(tokens), tokens))

    encode_plus_tokens = tokenizer.encode_plus(
        the_input.text,
        pad_to_max_length=True,
        max_length=max_seq_length,
        truncation=True
    )
```
# 3. Model Train and Tuning
# 3-1. BERTなどのNLPモデル
The NLP algorithm called Word2vec takes a sentence from Wikipedia, etc., focuses on a specific word, and uses the words before and after that word as training data to calculate the weight to be given to the middle layer. The space that holds these weights as word-specific vectors is called a corpus, and it functions as a database that collects natural language sentences and usage on a large scale and organizes them so that they can be searched by computer. BERT and Chat-GPT use a method called attention to calculate this weight. The model cannot be created using the computational resources of one or two GPU machines.<br>
Word2vecと呼ばれるNLPアルゴリズムでは、Wikipediaなどから文章を持ってきて、特定の単語に注目したときに、その単語の前後にある単語を教師データとして中間層に与える重みを計算する。
この重みを単語固有のベクトルとして保持している空間をコーパスと呼び、自然言語の文章や使い方を大規模に収集しコンピュータで検索できるよう整理されたデータベースとして機能する。
この重みを計算する手法に、アテンションと呼ばれる手法を使ったものが、BERTやChat-GPTである。1台や2台程度のGPUマシンの計算リソースでは当該モデルを作ることはできない。<br>

例：<br>
こたつ　で　みかん　を　食べる<br>
　入力値：みかん<br>
　教師データ：食べる<br>
<br>

After all, pre-traing is learning to acquire general-purpose linguistic knowledge by learning how to use words based on a large-scale corpus using natural language processing such as BERT.<br>
結局、事前学習とは、BERTなどの自然言語処理で大規模なコーパスをもとに言葉の使い方を学習することで、汎用的な言語知識(みかんは食べられるものであるという常識)を獲得する学習である。

>![Word2vec.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/Word2vec.drawio.png)

# 3-2. SageMaker JumpStart (Fine Tuning)
Fine-tune using the BERT embedding from the review_body text already generated in 2-4 and create a custom classifier that predicts star_rating as shown in the figure below.<br>
既に2-4で生成済みのreview_bodyテキストからのBERT埋め込みを使用してファインチューニングし、以下図に示すようにstar_ratingを予測するカスタム分類器を作成する。
>![StarRating.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/StarRating.drawio.png)

This additional learning of pre-trained models such as BERT and Chat-GPT using downstream task datasets is called fine-tuning. Task-specific classifiers (heads) have a simple structure consisting of a small number of parameters and require fewer computational resources than those required for pre-training.<br>
このように、BERTやChat-GPTのように事前学習されたモデルを下流タスクのデータセットで微調整することをファインチューニングと呼ぶ。タスク固有の分類器(ヘッド)は少量のパラメータで構成される単純な構造となり、事前学習で求められる計算リソースに比べて少なくて済む。

>![Vector.png](https://github.com/developer-onizuka/Diagrams/blob/main/MachineLearningOnAWS/Vector.drawio.png)

# 3-3. On My Workstation (Fine Tuning)
```
$ git clone https://github.com/developer-onizuka/MachineLearningOnAWS
$ cd MachineLearning
$ virsh pool-start data
$ vagrant up --provider=libvirt
$ vagrant ssh
```
```
$ sudo docker run --name nvidia-driver -itd --rm --privileged --pid=host -v /run/nvidia:/run/nvidia:shared -v /var/log:/var/log  nvcr.io/nvidia/driver:535.129.03-ubuntu20.04
$ sudo docker logs -f nvidia-driver
$ sudo docker run -it --rm --gpus all -v /home/vagrant:/mnt --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name tensorflow nvcr.io/nvidia/tensorflow:23.07-tf2-py3
```
Run BERT-embedding-from-text.ipynb for Data Preparation and Fine-Tuning.ipynb for Fine Tuning.
```
+---------------------------------------------------------------------------------------+
Mon Jan  8 13:27:12 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Quadro P4000                   On  | 00000000:03:00.0 Off |                  N/A |
| 51%   53C    P0              72W / 105W |   7422MiB /  8192MiB |     66%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+

```

# 4. Deploy and Monitoring




