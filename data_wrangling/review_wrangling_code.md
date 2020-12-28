```pyspark
filenames = ['asH', 'aus', 'bos', 'bro', 'cam', 'chicago', 'clb',
             'clk', 'dc', 'den', 'haw', 'jer', 'la', 'msa', 'nash',
             'norl', 'nyc', 'oslo', 'pacf', 'ptl', 'rhode', 'salem',
             'sanmateo', 'sd', 'seattle', 'sf', 'stclara', 'stcruz'] 

paths = ['s3://projectyfl12/reviews/reviews_{}.csv'.format(name)
         for name in filenames]
paths
```


    VBox()


    Starting Spark application



<table>
<tr><th>ID</th><th>YARN Application ID</th><th>Kind</th><th>State</th><th>Spark UI</th><th>Driver log</th><th>Current session?</th></tr><tr><td>7</td><td>application_1607541500596_0008</td><td>pyspark</td><td>idle</td><td></td><td></td><td>✔</td></tr></table>



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    SparkSession available as 'spark'.



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    ['s3://projectyfl12/reviews/reviews_asH.csv', 's3://projectyfl12/reviews/reviews_aus.csv', 's3://projectyfl12/reviews/reviews_bos.csv', 's3://projectyfl12/reviews/reviews_bro.csv', 's3://projectyfl12/reviews/reviews_cam.csv', 's3://projectyfl12/reviews/reviews_chicago.csv', 's3://projectyfl12/reviews/reviews_clb.csv', 's3://projectyfl12/reviews/reviews_clk.csv', 's3://projectyfl12/reviews/reviews_dc.csv', 's3://projectyfl12/reviews/reviews_den.csv', 's3://projectyfl12/reviews/reviews_haw.csv', 's3://projectyfl12/reviews/reviews_jer.csv', 's3://projectyfl12/reviews/reviews_la.csv', 's3://projectyfl12/reviews/reviews_msa.csv', 's3://projectyfl12/reviews/reviews_nash.csv', 's3://projectyfl12/reviews/reviews_norl.csv', 's3://projectyfl12/reviews/reviews_nyc.csv', 's3://projectyfl12/reviews/reviews_oslo.csv', 's3://projectyfl12/reviews/reviews_pacf.csv', 's3://projectyfl12/reviews/reviews_ptl.csv', 's3://projectyfl12/reviews/reviews_rhode.csv', 's3://projectyfl12/reviews/reviews_salem.csv', 's3://projectyfl12/reviews/reviews_sanmateo.csv', 's3://projectyfl12/reviews/reviews_sd.csv', 's3://projectyfl12/reviews/reviews_seattle.csv', 's3://projectyfl12/reviews/reviews_sf.csv', 's3://projectyfl12/reviews/reviews_stclara.csv', 's3://projectyfl12/reviews/reviews_stcruz.csv']


```pyspark
# df1 = spark.read.option("header", True).csv("s3://projectyfl12/reviews/reviews_asH.csv")
# df2 = spark.read.option("header", True).csv("s3://projectyfl12/listings/listings_asH.csv")

dfs = []
for p in paths:
    df = spark.read.option("header", True).csv(p)
    df = df.filter(df.comments.isNotNull())
    dfs.append(df)

```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…




```pyspark
len(dfs)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    28


```pyspark
dfs[0].show(5)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    +----------+------+----------+-----------+-------------+--------------------+
    |listing_id|    id|      date|reviewer_id|reviewer_name|            comments|
    +----------+------+----------+-----------+-------------+--------------------+
    |     38585|129120|2010-10-28|      55877|      Ritchie|Evelyne is an acc...|
    |     38585|147273|2010-11-30|     279973|        Cathy|Evelyne was very ...|
    |     38585|198797|2011-03-14|     411638|          N/A|"I really enjoyed...|
    |     38585|201932|2011-03-17|     441855|         Bill|Very gracious hos...|
    |     38585|341616|2011-06-28|     657560|       Joakim|Evelyn was very f...|
    +----------+------+----------+-----------+-------------+--------------------+
    only showing top 5 rows


```pyspark
dfs[1].show(5)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    +----------+-------+----------+-----------+-------------+--------------------+
    |listing_id|     id|      date|reviewer_id|reviewer_name|            comments|
    +----------+-------+----------+-----------+-------------+--------------------+
    |      2265|    963|2009-03-17|       7538|        Niall|I stayed here dur...|
    |      2265|   1057|2009-03-22|      10029|      Michael|Great place, clos...|
    |      2265| 200418|2011-03-16|      61677|       Gustaf|We had a great ti...|
    |      2265|1001630|2012-03-15|    1523753|         Noah|We had a great st...|
    |      2265|1016390|2012-03-19|    1547660|      Melissa|I arrived late in...|
    +----------+-------+----------+-----------+-------------+--------------------+
    only showing top 5 rows


```pyspark
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, concat_ws
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
def wrangle_comment(df):
    REGEX = '[,\\-!.~$"]'
    df = df.withColumn('comments', regexp_replace(df.comments, REGEX, ' '))
    df = Tokenizer(inputCol="comments", outputCol="comments_tokens").transform(df)
    stopwords = StopWordsRemover()
    stopwords = stopwords.setInputCol('comments_tokens').setOutputCol('comments_words')
    df = stopwords.transform(df)
    df = df.withColumn("comments_words",
       concat_ws(" ",col("comments_words")))
    return df
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark

dfss = []

for df in dfs:
    df = wrangle_comment(df)
    dfss.append(df)
    
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark

df_u = dfss[0]
for df in dfss[1:]:
    df_u = df_u.union(df)
    
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_u.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    8542091


```pyspark
df_u.show()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    +----------+-------+----------+-----------+-------------+--------------------+--------------------+--------------------+
    |listing_id|     id|      date|reviewer_id|reviewer_name|            comments|     comments_tokens|      comments_words|
    +----------+-------+----------+-----------+-------------+--------------------+--------------------+--------------------+
    |     38585| 129120|2010-10-28|      55877|      Ritchie|Evelyne is an acc...|[evelyne, is, an,...|evelyne accommoda...|
    |     38585| 147273|2010-11-30|     279973|        Cathy|Evelyne was very ...|[evelyne, was, ve...|evelyne welcoming...|
    |     38585| 198797|2011-03-14|     411638|          N/A| I really enjoyed...|[, i, really, enj...| really enjoyed e...|
    |     38585| 201932|2011-03-17|     441855|         Bill|Very gracious hos...|[very, gracious, ...|gracious host hel...|
    |     38585| 341616|2011-06-28|     657560|       Joakim|Evelyn was very f...|[evelyn, was, ver...|evelyn friendly e...|
    |     38585| 369937|2011-07-12|     792195|    Gabrielle| Evelyne was a gr...|[, evelyne, was, ...| evelyne gracious...|
    |     38585| 376614|2011-07-16|     768992|        Horst|If there were mor...|[if, there, were,...|people like evely...|
    |     38585| 403463|2011-07-28|     819690|       Sathya|Evelyne and her s...|[evelyne, and, he...|evelyne son trist...|
    |     38585| 488018|2011-08-30|     936491|      Timothy|Evelyne was a gre...|[evelyne, was, a,...|evelyne great hos...|
    |     38585| 627253|2011-10-14|     481222|        Sandy|Evelyne is simply...|[evelyne, is, sim...|evelyne simply on...|
    |     38585| 638260|2011-10-17|    1111462|      Andreas|We had a great ti...|[we, had, a, grea...|great time clean ...|
    |     38585| 682043|2011-11-01|     598479|       Seanna|Evelyne was a kin...|[evelyne, was, a,...|evelyne kind wond...|
    |     38585| 718841|2011-11-14|    1368414|        Cindy|My husband and I ...|[my, husband, and...|husband wonderful...|
    |     38585| 747593|2011-11-27|    1409277|        Julia|Evelyne was a won...|[evelyne, was, a,...|evelyne wonderful...|
    |     38585| 975787|2012-03-06|    1695383|       Sheila|Evelyne was a gre...|[evelyne, was, a,...|evelyne great hos...|
    |     38585|1327551|2012-05-22|    1676274|         Paul|friendly host   h...|[friendly, host, ...|friendly host   h...|
    |     38585|1366093|2012-05-28|    2363328|       Robert|Our stay with Eve...|[our, stay, with,...|stay evelyne wond...|
    |     38585|1386767|2012-05-30|     147135|        Leora|Evelyne's home is...|[evelyne's, home,...|evelyne's home clean|
    |     38585|1750474|2012-07-19|      13139|         Mike|Great experience ...|[great, experienc...|great experience ...|
    |     38585|1952739|2012-08-11|    3113338|        Emily|My sister and I v...|[my, sister, and,...|sister much enjoy...|
    +----------+-------+----------+-----------+-------------+--------------------+--------------------+--------------------+
    only showing top 20 rows


```pyspark
df_u[['comments_words', 'listing_id', 'id']].write.csv('commentsall.csv')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    No module named 'mrjob'
    Traceback (most recent call last):
    ModuleNotFoundError: No module named 'mrjob'
    



```pyspark

```
