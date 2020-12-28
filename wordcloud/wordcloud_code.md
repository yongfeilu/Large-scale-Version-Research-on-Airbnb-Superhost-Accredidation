```bash
%%bash
apt-get install openjdk-8-jdk-headless -qq > /dev/null
wget -q https://downloads.apache.org/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz
tar -xzf spark-2.4.7-bin-hadoop2.7.tgz
pip install pyspark findspark
```

    Collecting pyspark
      Downloading https://files.pythonhosted.org/packages/f0/26/198fc8c0b98580f617cb03cb298c6056587b8f0447e20fa40c5b634ced77/pyspark-3.0.1.tar.gz (204.2MB)
    Collecting findspark
      Downloading https://files.pythonhosted.org/packages/fc/2d/2e39f9a023479ea798eed4351cd66f163ce61e00c717e03c37109f00c0f2/findspark-1.4.2-py2.py3-none-any.whl
    Collecting py4j==0.10.9
      Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)
    Building wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py): started
      Building wheel for pyspark (setup.py): finished with status 'done'
      Created wheel for pyspark: filename=pyspark-3.0.1-py2.py3-none-any.whl size=204612243 sha256=aee167caaad4f750d31091b8c2f3cdb287319850eae964594f6b9adf4b05d15b
      Stored in directory: /root/.cache/pip/wheels/5e/bd/07/031766ca628adec8435bb40f0bd83bb676ce65ff4007f8e73f
    Successfully built pyspark
    Installing collected packages: py4j, pyspark, findspark
    Successfully installed findspark-1.4.2 py4j-0.10.9 pyspark-3.0.1



```python
import os
os.environ["SPARK_HOME"] = "/content/spark-2.4.7-bin-hadoop2.7"
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
```


```python
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, concat_ws
from pyspark.sql import SQLContext
from pyspark.sql.types import *
```


```python
def wrangle_reviews(df):

  mySchema = StructType([StructField("comments", StringType(), True)])

  spark_df = spark.createDataFrame(df, schema=mySchema)
  spark_df = spark_df.filter(spark_df.comments.isNotNull())
  REGEX = '[,\\-!.~$]'
  spark_df = spark_df.withColumn('comments', regexp_replace(spark_df.comments, REGEX, ' '))
  spark_df = Tokenizer(inputCol="comments", outputCol="comments_tokens").transform(spark_df)
  stopwords = StopWordsRemover()
  stopwords = stopwords.setInputCol('comments_tokens').setOutputCol('comments_words')
  spark_df = stopwords.transform(spark_df)
  df = spark_df.withColumn("comments_words",
   concat_ws(" ",col("comments_words")))
  return df[['comments_words']]
```


```python
import pandas as pd
df_high = pd.read_csv("chigh.csv")
```


```python
df_low = pd.read_csv("clow.csv")
```


```python
df_high = wrangle_reviews(df_high)
```


```python
df_low = wrangle_reviews(df_low)
```


```python
df_high.write.csv('comments_high.csv')
```


```python
df_low.write.csv('comments_low.csv')
```


```python
lc = pd.read_csv('lc.csv', sep='\\t', header=None)
hc = pd.read_csv('hc.csv', sep='\\t', header=None)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      """Entry point for launching an IPython kernel.
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      



```python
hd= {}
for index, row in hc.iterrows():
  hd[row[1].strip('''"''')] = row[0]
```


```python
ld= {}
for index, row in lc.iterrows():
  ld[row[1].strip('''"''')] = row[0]
```


```python
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud 

wc = WordCloud(background_color="white",width=1000,height=1000, max_words=100,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(hd)
plt.imshow(wc)
```




    <matplotlib.image.AxesImage at 0x7f2ed3c70160>




![png](output_13_1.png)



```python
wc = WordCloud(background_color="white",width=3000,height=3000, max_words=100,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(ld)
plt.imshow(wc)
```




    <matplotlib.image.AxesImage at 0x7f2ed64e0748>




![png](output_14_1.png)



```python

```
