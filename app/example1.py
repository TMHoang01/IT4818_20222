
from pyspark.sql import SparkSession

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, mean, sum


spark = SparkSession.builder.appName("best_params").master('spark://9b48a6b6138c:7077').getOrCreate()

# Đường dẫn đến file CSV
file_path = '/app/data.csv'

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql.functions import to_date

# Khai báo kiểu dữ liệu cho từng cột
schema = StructType([
    StructField("product_id", StringType()),
    StructField("product_category_name", StringType()),
    StructField("month_year", TimestampType()),
    StructField("qty", IntegerType()),
    StructField("total_price", FloatType()),
    StructField("freight_price", FloatType()),
    StructField("unit_price", FloatType()),
    StructField("product_name_length", IntegerType()),
    StructField("product_description_length", IntegerType()),
    StructField("product_photos_qty", IntegerType()),
    StructField("product_weight_g", IntegerType()),
    StructField("product_score", FloatType()),
    StructField("customers", IntegerType()),
    StructField("weekday", IntegerType()),
    StructField("weekend", IntegerType()),
    StructField("holiday", IntegerType()),
    StructField("month", IntegerType()),
    StructField("year", IntegerType()),
    StructField("s", FloatType()),
    StructField("volume", IntegerType()),
    StructField("comp_1", FloatType()),
    StructField("ps1", FloatType()),
    StructField("fp1", FloatType()),
    StructField("comp_2", FloatType()),
    StructField("ps2", FloatType()),
    StructField("fp2", FloatType()),
    StructField("comp_3", FloatType()),
    StructField("ps3", FloatType()),
    StructField("fp3", FloatType()),
    StructField("lag_price", FloatType())
])


# Đọc dữ liệu từ file CSV vào DataFrame PySpark
df = spark.read.format("csv").option("header", "true").option("timestampFormat", "dd-MM-yyyy").schema(schema).load(file_path)



# Xác định cột đầu vào (X) và cột đầu ra (y)
x_cols = ['qty', 'total_price', 'freight_price', 'product_score', 'product_weight_g', 'customers', 
          'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume', 'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

y_col = 'unit_price'




from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)


# Chọn các cột đặc trưng
feature_cols = x_cols 

# Tạo VectorAssembler để chuyển đổi các cột đặc trưng thành một cột features
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Áp dụng VectorAssembler để tạo cột features trong DataFrame
train_data = assembler.transform(train_data)

# Khởi tạo mô hình RandomForestRegressor
model = RandomForestRegressor(labelCol='unit_price')

# Định nghĩa các giá trị tham số cần thử
param_grid = ParamGridBuilder() \
    .addGrid(model.numTrees, [10, 30, 50, 70 ,100, 125]) \
    .addGrid(model.maxDepth, [5, 10, 15, 20]) \
    .addGrid(model.minInstancesPerNode, [1, 3, 5]) \
    .build()


# Khởi tạo Pipeline với mô hình và các bước xử lý dữ liệu khác (nếu có)
pipeline = Pipeline(stages=[model])

# Khởi tạo CrossValidator với Pipeline, Evaluator và ParamGrid
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=RegressionEvaluator(labelCol='unit_price', metricName='mse'),
                          numFolds=3)

# Huấn luyện mô hình và tinh chỉnh tham số trên dữ liệu
cvModel = crossval.fit(train_data)

# Lấy mô hình tốt nhất và các tham số tương ứng
best_model = cvModel.bestModel
best_params = best_model.stages[0].extractParamMap()

print("Best parameters:")
for key, value in best_params.items():
    print(key, "=", value)


# Tạo cột features cho test_data
test_data = assembler.transform(test_data)

# Đánh giá mô hình trên tập kiểm tra
predictions = best_model.transform(test_data)

# Tính toán độ đo RMSE

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol='unit_price', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print("RMSE on test data = %g" % rmse)
evaluator = RegressionEvaluator(labelCol='unit_price', metricName="r2")
# Tính R2 score trên dữ liệu huấn luyện
r2_score = evaluator.evaluate(predictions)
print(f"R2 score: {r2_score}")


# Ghi kết quả vào file
with open('/app/best_params2.txt', 'w') as f:
    f.write('count: {}\n'.format(df.count()))
    for key, value in best_params.items():
        f.write('{}: {}\n'.format(key, value))
    f.write('RMSE: {}\n'.format(rmse))
    f.write('R2: {}'.format(r2_score))

best_model.stages[0].save("/app/model1")


spark.stop()
