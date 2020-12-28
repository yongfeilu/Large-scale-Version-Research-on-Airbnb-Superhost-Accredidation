```pyspark
df = spark.read.csv('s3://projectyfl12/listings/ls.csv',
                      sep=",",
                      header=True,
                      inferSchema=True)


```


    VBox()


    Starting Spark application



<table>
<tr><th>ID</th><th>YARN Application ID</th><th>Kind</th><th>State</th><th>Spark UI</th><th>Driver log</th><th>Current session?</th></tr><tr><td>4</td><td>application_1607541500596_0005</td><td>pyspark</td><td>idle</td><td><a target="_blank" href="http://ip-172-31-21-200.ec2.internal:20888/proxy/application_1607541500596_0005/">Link</a></td><td><a target="_blank" href="http://ip-172-31-17-75.ec2.internal:8042/node/containerlogs/container_1607541500596_0005_01_000001/livy">Link</a></td><td>✔</td></tr></table>



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    SparkSession available as 'spark'.



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df = df.withColumn("reviews_per_month", df.reviews_per_month.cast("double"))
df = df.withColumn("accommodates", df.accommodates.cast("integer"))
df = df.withColumn('super_host', (df.host_is_superhost == 't').cast("integer"))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df = df.dropna()
df.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- _c0: string (nullable = true)
     |-- host_is_superhost: string (nullable = true)
     |-- city: string (nullable = true)
     |-- price: double (nullable = true)
     |-- room_type: string (nullable = true)
     |-- latitude: double (nullable = true)
     |-- longitude: double (nullable = true)
     |-- reviews_per_month: double (nullable = true)
     |-- number_of_reviews: double (nullable = true)
     |-- cancellation_policy: string (nullable = true)
     |-- security_deposit: double (nullable = true)
     |-- cleaning_fee: double (nullable = true)
     |-- beds: double (nullable = true)
     |-- bedrooms: double (nullable = true)
     |-- bathrooms: double (nullable = true)
     |-- accommodates: integer (nullable = true)
     |-- host_response_time: double (nullable = true)
     |-- host_identity_verified: string (nullable = true)
     |-- availability_30: double (nullable = true)
     |-- instant_bookable: string (nullable = true)
     |-- review_scores_rating: double (nullable = true)
     |-- host_response_rate: double (nullable = true)
     |-- occupancy_rate: double (nullable = true)
     |-- revenue: double (nullable = true)
     |-- host_time: integer (nullable = true)
     |-- is_TV: integer (nullable = true)
     |-- is_Wifi: integer (nullable = true)
     |-- amenities_number: integer (nullable = true)
     |-- super_host: integer (nullable = true)


```pyspark
cols = ['super_host','occupancy_rate','host_time','host_response_time',
    'reviews_per_month','number_of_reviews','review_scores_rating',
    'cancellation_policy','cleaning_fee','security_deposit','beds',
    'bedrooms','bathrooms','accommodates', 'availability_30',
    'instant_bookable', 'amenities_number', 'room_type', 'price']
df_m = df[cols]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
indexer_ib = StringIndexer(inputCol="instant_bookable",outputCol="instant_bookable_idx")
indexer_cp = StringIndexer(inputCol="cancellation_policy",outputCol="cancellation_policy_idx")
indexer_rt = StringIndexer(inputCol="room_type",outputCol="room_type_idx")

features = ['super_host','occupancy_rate','host_time','host_response_time',
            'reviews_per_month','number_of_reviews','review_scores_rating',
            'cancellation_policy_idx','cleaning_fee','security_deposit','beds',
            'bedrooms','bathrooms','accommodates', 'availability_30',
            'instant_bookable_idx', 'amenities_number', 'room_type_idx']



assembler = VectorAssembler(inputCols=features, outputCol='features')

regression = LinearRegression(featuresCol='features', labelCol='price')
evaluator = RegressionEvaluator(labelCol='price')

train, test = df_m.randomSplit([0.7, 0.3])

pipeline = Pipeline(stages=[indexer_ib, indexer_cp, indexer_rt, assembler, regression])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark

```


```pyspark
params = ParamGridBuilder() \
        .addGrid(regression.fitIntercept, [True, False]) \
        .addGrid(regression.regParam, [0.001, 0.01, 0.1, 1, 10]) \
        .addGrid(regression.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]) \
        .build()
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator,
                    numFolds=10, seed=13)


```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
print ('Number of models to be tested: ', len(params))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Number of models to be tested:  50


```pyspark
cv = cv.fit(train)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Exception in thread cell_monitor-16:
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.7/threading.py", line 926, in _bootstrap_inner
        self.run()
      File "/opt/conda/lib/python3.7/threading.py", line 870, in run
        self._target(*self._args, **self._kwargs)
      File "/opt/conda/lib/python3.7/site-packages/awseditorssparkmonitoringwidget-1.0-py3.7.egg/awseditorssparkmonitoringwidget/cellmonitor.py", line 178, in cell_monitor
        job_binned_stages[job_id][stage_id] = all_stages[stage_id]
    KeyError: 3093
    



```pyspark

# cross-validated RMSE for each model
cv.avgMetrics
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    [511.2364851508609, 511.23645349119994, 511.2364946222264, 511.23649983715296, 511.23648672227876, 511.2364660223152, 511.2363950176856, 511.2363669667937, 511.23632712361393, 511.2363339415988, 511.2362759434803, 511.23582122618404, 511.2354578880783, 511.23526009991826, 511.2348961490163, 511.23449421313165, 511.2327112961652, 511.2340510902394, 511.23740856147936, 511.2447388597327, 511.22731395624527, 511.367518921711, 511.6392179832134, 512.0433826713373, 512.478562176409, 511.59398750373356, 511.5897518368468, 511.58495810468065, 511.601929720063, 511.598417978205, 511.5939687705663, 511.5925498294628, 511.5937546100992, 511.6044863957752, 511.5986903686404, 511.59378256430097, 511.590718169136, 511.59235399600544, 511.6076705611904, 511.5926978267972, 511.592031730409, 511.5945040841657, 511.5923362033715, 511.58941632327355, 511.5998714659043, 511.58458769193635, 511.72228496584336, 511.9809977021667, 512.3736168032112, 512.7485447025009]


```pyspark
optimal_model = cv.bestModel

trainingSummary = optimal_model.stages[-1].summary
prediction = optimal_model.transform(test)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark

# Find RMSE (Root Mean Squared Error)
evaluator.evaluate(prediction)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    486.17103191240153


```pyspark
optimal_model.stages[-1].extractParamMap()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    {Param(parent='LinearRegression_a4dd4cc4ee61', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2)'): 2, Param(parent='LinearRegression_a4dd4cc4ee61', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty'): 0.0, Param(parent='LinearRegression_a4dd4cc4ee61', name='epsilon', doc='The shape parameter to control the amount of robustness. Must be > 1.0.'): 1.35, Param(parent='LinearRegression_a4dd4cc4ee61', name='featuresCol', doc='features column name'): 'features', Param(parent='LinearRegression_a4dd4cc4ee61', name='fitIntercept', doc='whether to fit an intercept term'): True, Param(parent='LinearRegression_a4dd4cc4ee61', name='labelCol', doc='label column name'): 'price', Param(parent='LinearRegression_a4dd4cc4ee61', name='loss', doc='The loss function to be optimized. Supported options: squaredError, huber. (Default squaredError)'): 'squaredError', Param(parent='LinearRegression_a4dd4cc4ee61', name='maxIter', doc='maximum number of iterations (>= 0)'): 100, Param(parent='LinearRegression_a4dd4cc4ee61', name='predictionCol', doc='prediction column name'): 'prediction', Param(parent='LinearRegression_a4dd4cc4ee61', name='regParam', doc='regularization parameter (>= 0)'): 10.0, Param(parent='LinearRegression_a4dd4cc4ee61', name='solver', doc='The solver algorithm for optimization. Supported options: auto, normal, l-bfgs. (Default auto)'): 'auto', Param(parent='LinearRegression_a4dd4cc4ee61', name='standardization', doc='whether to standardize the training features before fitting the model'): True, Param(parent='LinearRegression_a4dd4cc4ee61', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0)'): 1e-06}


```pyspark
optimal_model.stages[-1].coefficients
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    DenseVector([-12.8598, 0.7168, 0.0012, 33.546, 1.073, -0.2841, 1.1649, 27.6155, 0.861, 0.0766, -5.9315, 34.9668, 101.3423, 13.0131, 2.4593, 22.4888, -1.67, -53.7537])


```pyspark
optimal_model.stages[-1].intercept
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    -276.84633849682393


```pyspark
print("Coefficients: {} \n\nIntercept: {}".format(optimal_model.stages[-1].coefficients, optimal_model.stages[-1].intercept))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Coefficients: [-12.85983816323193,0.716828866680952,0.0012154850116938847,33.545951938043615,1.0729834378907455,-0.2841188078844449,1.1648892479888475,27.61554253675256,0.8610239949957893,0.07660420487336203,-5.931517486722173,34.96682735727024,101.34232239634251,13.01312811320191,2.4592719124352107,22.488769961425024,-1.6700061133883066,-53.753701419365946] 
    
    Intercept: -276.84633849682393


```pyspark
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    numIterations: 1
    objectiveHistory: [0.0]
    +-------------------+
    |          residuals|
    +-------------------+
    | 310.06439408973847|
    |-107.28306683296682|
    |  93.54922147480306|
    | -514.3166213757847|
    |  21.00346107234111|
    |  139.3976450538206|
    |  115.2465342830559|
    | -214.6147820906674|
    | 116.70755566169476|
    |-108.58836978088425|
    |-181.23697337564647|
    |-257.02071873566456|
    | -195.5423205688113|
    |-132.74235156021462|
    | -219.7995192009521|
    |-156.82725705235055|
    |  1180.577514373283|
    | -452.3465076370004|
    |-209.43868814922553|
    | -81.70528358891283|
    +-------------------+
    only showing top 20 rows
    
    RMSE: 521.214147
    r2: 0.180609


```pyspark

```


```pyspark

```


```pyspark

```


```pyspark

```


```pyspark

```


```pyspark

```


```pyspark

```


```pyspark
df_r = spark.read.csv('s3://projectyfl12/listings/ls.csv',
                      sep=",",
                      header=True,
                      inferSchema=True)


```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_r = df_r.withColumn("reviews_per_month", df_r.reviews_per_month.cast("double"))
df_r = df_r.withColumn("accommodates", df_r.accommodates.cast("integer"))
df_r = df_r.withColumn('super_host', (df_r.host_is_superhost == 't').cast("integer"))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_r = df_r.dropna()
df_r.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- _c0: string (nullable = true)
     |-- host_is_superhost: string (nullable = true)
     |-- city: string (nullable = true)
     |-- price: double (nullable = true)
     |-- room_type: string (nullable = true)
     |-- latitude: double (nullable = true)
     |-- longitude: double (nullable = true)
     |-- reviews_per_month: double (nullable = true)
     |-- number_of_reviews: double (nullable = true)
     |-- cancellation_policy: string (nullable = true)
     |-- security_deposit: double (nullable = true)
     |-- cleaning_fee: double (nullable = true)
     |-- beds: double (nullable = true)
     |-- bedrooms: double (nullable = true)
     |-- bathrooms: double (nullable = true)
     |-- accommodates: integer (nullable = true)
     |-- host_response_time: double (nullable = true)
     |-- host_identity_verified: string (nullable = true)
     |-- availability_30: double (nullable = true)
     |-- instant_bookable: string (nullable = true)
     |-- review_scores_rating: double (nullable = true)
     |-- host_response_rate: double (nullable = true)
     |-- occupancy_rate: double (nullable = true)
     |-- revenue: double (nullable = true)
     |-- host_time: integer (nullable = true)
     |-- is_TV: integer (nullable = true)
     |-- is_Wifi: integer (nullable = true)
     |-- amenities_number: integer (nullable = true)
     |-- super_host: integer (nullable = true)


```pyspark

indexer_ib = StringIndexer(inputCol="instant_bookable",outputCol="instant_bookable_idx")
indexer_cp = StringIndexer(inputCol="cancellation_policy",outputCol="cancellation_policy_idx")
indexer_rt = StringIndexer(inputCol="room_type",outputCol="room_type_idx")

features = ['super_host','occupancy_rate','price','host_time','reviews_per_month',
            'number_of_reviews', 'instant_bookable_idx', 'review_scores_rating',
            'host_response_rate','cancellation_policy_idx','security_deposit','cleaning_fee',
            'beds','bedrooms','bathrooms','accommodates','amenities_number','room_type_idx']



assembler_r = VectorAssembler(inputCols=features, outputCol='features')

regression_r = LinearRegression(featuresCol='features', labelCol='revenue')
evaluator = RegressionEvaluator(labelCol='price')

train_r, test_r = df_r.randomSplit([0.7, 0.3])

pipeline_r = Pipeline(stages=[indexer_ib, indexer_cp, indexer_rt, assembler_r, regression_r])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
params_r = ParamGridBuilder() \
        .addGrid(regression_r.fitIntercept, [True, False]) \
        .addGrid(regression_r.regParam, [0.001, 0.01, 0.1, 1, 10]) \
        .addGrid(regression_r.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]) \
        .build()
cv_r = CrossValidator(estimator=pipeline_r,
                    estimatorParamMaps=params_r,
                    evaluator=evaluator,
                    numFolds=10, seed=13)

print ('Number of models to be tested: ', len(params_r))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Number of models to be tested:  50


```pyspark
cv_r = cv_r.fit(train_r)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Exception in thread cell_monitor-15:
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.7/threading.py", line 926, in _bootstrap_inner
        self.run()
      File "/opt/conda/lib/python3.7/threading.py", line 870, in run
        self._target(*self._args, **self._kwargs)
      File "/opt/conda/lib/python3.7/site-packages/awseditorssparkmonitoringwidget-1.0-py3.7.egg/awseditorssparkmonitoringwidget/cellmonitor.py", line 178, in cell_monitor
        job_binned_stages[job_id][stage_id] = all_stages[stage_id]
    KeyError: 3291
    



```pyspark
# cross-validated RMSE for each model
print('cross-validated RMSE for each model: \n', cv_r.avgMetrics)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    cross-validated RMSE for each model: 
     [6610.648176128331, 6610.648371271789, 6610.647953302593, 6610.647976702824, 6610.626561763009, 6610.640650430053, 6610.608422166411, 6610.6032449238155, 6610.642360558608, 6610.642965460744, 6610.565394808958, 6610.4211112175435, 6610.440149906541, 6610.516752330724, 6610.5636601982615, 6609.812974733553, 6609.650693379812, 6609.488671669769, 6609.326722590367, 6609.165225872179, 6602.302352191537, 6600.695604156782, 6599.114364444082, 6597.559222634363, 6596.0215885561765, 6609.94306962874, 6609.818558306357, 6609.911651997197, 6610.064283764658, 6609.794347838762, 6609.93554135233, 6608.4736197544535, 6610.6201585305935, 6611.039014225857, 6610.024105017444, 6609.860259947966, 6609.344025140268, 6610.281317465649, 6610.202602157487, 6610.574373421582, 6609.107581842473, 6607.763936419688, 6607.971046880065, 6608.5880461325305, 6609.753569129765, 6601.594359752109, 6600.171530432977, 6598.720930539596, 6597.236710020893, 6595.282964317307]


```pyspark
optimal_model_r = cv_r.bestModel

trainingSummary_r = optimal_model_r.stages[-1].summary
prediction_r = optimal_model_r.transform(test_r)
# Find RMSE (Root Mean Squared Error)
print('RMSE of optimal regression model on testing dataset: ', evaluator.evaluate(prediction_r))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    RMSE of optimal regression model on testing dataset:  6208.8406840790985


```pyspark
optimal_model_r.stages[-1].extractParamMap()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    {Param(parent='LinearRegression_39ad701627c8', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2)'): 2, Param(parent='LinearRegression_39ad701627c8', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty'): 1.0, Param(parent='LinearRegression_39ad701627c8', name='epsilon', doc='The shape parameter to control the amount of robustness. Must be > 1.0.'): 1.35, Param(parent='LinearRegression_39ad701627c8', name='featuresCol', doc='features column name'): 'features', Param(parent='LinearRegression_39ad701627c8', name='fitIntercept', doc='whether to fit an intercept term'): False, Param(parent='LinearRegression_39ad701627c8', name='labelCol', doc='label column name'): 'revenue', Param(parent='LinearRegression_39ad701627c8', name='loss', doc='The loss function to be optimized. Supported options: squaredError, huber. (Default squaredError)'): 'squaredError', Param(parent='LinearRegression_39ad701627c8', name='maxIter', doc='maximum number of iterations (>= 0)'): 100, Param(parent='LinearRegression_39ad701627c8', name='predictionCol', doc='prediction column name'): 'prediction', Param(parent='LinearRegression_39ad701627c8', name='regParam', doc='regularization parameter (>= 0)'): 10.0, Param(parent='LinearRegression_39ad701627c8', name='solver', doc='The solver algorithm for optimization. Supported options: auto, normal, l-bfgs. (Default auto)'): 'auto', Param(parent='LinearRegression_39ad701627c8', name='standardization', doc='whether to standardize the training features before fitting the model'): True, Param(parent='LinearRegression_39ad701627c8', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0)'): 1e-06}


```pyspark
print("Coefficients: {} \n\nIntercept: {}".format(optimal_model_r.stages[-1].coefficients, optimal_model_r.stages[-1].intercept))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Coefficients: [157.99207335089173,40.96882326645172,11.600916952926584,0.014596201612220847,423.11110895227324,-5.053031275758151,0.0,-20.361859422513422,1.150637501012787,-9.640657494422525,-0.05118225980485538,-4.927459470849323,67.1926071262587,-102.06788473828274,-257.785708112987,0.0,3.885216770900697,9.447240808884839] 
    
    Intercept: 0.0


```pyspark
features = ['super_host','occupancy_rate','price','host_time','reviews_per_month',
            'number_of_reviews', 'instant_bookable_idx', 'review_scores_rating',
            'host_response_rate','cancellation_policy_idx','security_deposit','cleaning_fee',
            'beds','bedrooms','bathrooms','accommodates','amenities_number','room_type_idx']
```


```pyspark
print("numIterations: %d" % trainingSummary_r.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary_r.objectiveHistory))
trainingSummary_r.residuals.show()
print("RMSE: %f" % trainingSummary_r.rootMeanSquaredError)
print("r2: %f" % trainingSummary_r.r2)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    numIterations: 1
    objectiveHistory: [0.0]

    numIterations: 57
    objectiveHistory: [0.5, 0.45416279466833914, 0.1956108082064838, 0.1600156063958606, 0.14928284948740536, 0.14867341486544156, 0.1484826534470529, 0.1484163135418803, 0.14839946052013522, 0.14839433489942314, 0.14839121070997532, 0.1483880358647825, 0.1483823260820837, 0.14837972421355022, 0.1483782977346422, 0.14837821531113693, 0.1483781659182514, 0.14837802108815024, 0.148377994834962, 0.1483779507863232, 0.1483779395274186, 0.14837792181942072, 0.1483779144412731, 0.14837791262932418, 0.14837789454895545, 0.1483778923818248, 0.14837788961318107, 0.14837788854854803, 0.14837788797016147, 0.14837788761778245, 0.14837788741071747, 0.14837788730793736, 0.14837788729588336, 0.14837788728836285, 0.14837788728476325, 0.14837788728289644, 0.14837788728200965, 0.14837788728143458, 0.14837788728110007, 0.14837788728079135, 0.1483778872805335, 0.14837788728022613, 0.14837788727983914, 0.1483778872794631, 0.14837788727921916, 0.14837788727912682, 0.14837788727902276, 0.14837788727890322, 0.14837788727879164, 0.14837788727874926, 0.14837788727869924, 0.14837788727866424, 0.1483778872786395, 0.1483778872786265, 0.14837788727861426, 0.1483778872786068, 0.14837788727860163]    

    +-------------------+
    |          residuals|
    +-------------------+
    | 1076.4759806706227|
    |  2054.633305015181|
    | -4945.479423201463|
    |  413.4860009902337|
    |-1132.2665247645982|
    |-1703.5993919732318|
    | -584.2164270373958|
    |  -1430.73317222271|
    | 326.95100356277817|
    | 1377.3627263348662|
    |  -802.306640040953|
    | -3962.170018739842|
    | 1191.0823691760745|
    |  1492.215005890751|
    |  909.6921424197521|
    |  792.8931406502552|
    | -434.3643040446084|
    |-1730.1087892446312|
    |  800.7901268861461|
    |  -519.652753225022|
    +-------------------+
    only showing top 20 rows

    
    RMSE: 4285.077007
    r2: 0.706575


```pyspark

```
