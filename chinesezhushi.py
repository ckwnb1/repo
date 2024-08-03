from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract,count,when,avg,lag,format_number,concat, lit,row_number,concat_ws,collect_list,expr
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType


# 创建SparkSession
spark = SparkSession.builder \
    .appName("Athlete Events Data Import") \
    .getOrCreate()

# 定义模式
schema = StructType([
    StructField("ID", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("Height", DoubleType(), True),
    StructField("Weight", DoubleType(), True),
    StructField("Team", StringType(), True),
    StructField("NOC", StringType(), True),
    StructField("Games", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Season", StringType(), True),
    StructField("City", StringType(), True),
    StructField("Sport", StringType(), True),
    StructField("Event", StringType(), True),
    StructField("Medal", StringType(), True)
])

# 文件路径
csv_file_path = "E:/hku/cloud cluster/ex4/dataset/athlete_events.csv"

# 读取CSV文件
spark_df = spark.read.csv(csv_file_path, schema=schema, header=True)


#筛选数据
spark_df = spark_df.filter((spark_df.Games.contains('Summer')) & (spark_df.Year >= 1956))

# 打印模式
spark_df.printSchema()

# 显示前五行数据
spark_df.show(5)


# 获取数值列的统计信息aa
spark_df.describe().show()







#eda1:Distribution and development trends of sports participation between male and female athletes\
#1.计算每一届奥运会分别有多少男性和女性运动员参赛   2.计算每届奥运会女性参赛者数量的增长率和比率



# 1. 按性别和年份分组，计算每组的运动员人数
gender_trend_df = spark_df.groupBy("Sex", "Year").agg(count("ID").alias("Participant_Count"))

# 2. 对结果进行排序
sorted_gender_trend_df = gender_trend_df.orderBy("Year", "Sex")

# 查看前21行
sorted_gender_trend_df.show(20)


# 1. 按性别和年份分组，计算每组的运动员人数
gender_trend_df = spark_df.groupBy("Sex", "Year").agg(
    count("ID").alias("Participant_Count")
)

# 2. 计算年增长率和参与比例
window_spec = Window.partitionBy("Sex").orderBy("Year")
gender_trend_df = gender_trend_df.withColumn(
    "Prev_Year_Participant", lag("Participant_Count").over(window_spec)
).withColumn(
    "Growth_Rate", 
    ((col("Participant_Count") - col("Prev_Year_Participant")) / col("Prev_Year_Participant") * 100).cast("decimal(10,3)")
).withColumn(
    "Growth_Rate", concat(col("Growth_Rate"), lit("%"))
)

# 3. 计算每年的总参与人数和参与比例
total_participants_df = spark_df.groupBy("Year").agg(count("ID").alias("Total_Participants"))
gender_ratio_df = gender_trend_df.join(total_participants_df, on="Year").withColumn(
    "Participation_Ratio", (col("Participant_Count") / col("Total_Participants") * 100).cast("decimal(10,3)")
).withColumn(
    "Participation_Ratio", concat(col("Participation_Ratio"), lit("%"))
)

# 捕获物理计划
gender_ratio_df.explain()

# 4. 展示所有数据
gender_ratio_df.orderBy("Year", "Sex").show(n=200000, truncate=False)






# EDA2：dentification of the sports infrastructure and traditional strong events of various countries
#.思路：随着时间的推进，各国奖牌总数的变化   2.筛选每项运动的主导国家。最后合并。比如运动1和运动2的主导国家都是法国，那么结果显示法国：运动1，运动2



medal_counts = spark_df.groupBy("Year", "NOC").agg(count(when(col("Medal").isNotNull(), 1)).alias("MedalCount"))

# 使用窗口函数找出每年奖牌总数排名前十的国

windowSpec = Window.partitionBy("Year").orderBy(col("MedalCount").desc())
top_10_countries = medal_counts.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") <= 10).drop("rank")

# 将结果显示为每年奖牌总数排名前十的国家和对应的奖牌数
result = top_10_countries.groupBy("Year").agg(concat_ws(", ", collect_list(concat_ws("-", col("NOC"), col("MedalCount")))).alias("Top10Countries"))

# 展示结果
result.show(20, truncate=False)


sport_leaders = spark_df.groupBy("NOC", "Sport").agg(count(when(col("Medal").isNotNull(), 1)).alias("MedalCount"))

# 找出每项运动的主导国家
windowSpec = Window.partitionBy("Sport").orderBy(col("MedalCount").desc())
sport_leaders = sport_leaders.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") == 1).drop("rank")

# 合并同一个国家的所有主导项目
from pyspark.sql.functions import collect_list

merged_sport_leaders = sport_leaders.groupBy("NOC").agg(concat_ws(", ", collect_list(concat_ws("-", col("Sport"), col("MedalCount")))).alias("DominantSports"))

# 展示结果
merged_sport_leaders.show(40, truncate=False)
sport_leaders.explain(True)








# EDA3：The Relationship Between Height Vs Weight Vs Age of Participants Across Sports
#分析各类运动的夺冠年龄，身高，体重分布，也就是说，把年龄从10-80，每5年分一个周期，10-15，15-20.....,身高，体重也如此，不过区间划分不同
#最后给出的结果类似于   ：     运动1       最多夺冠年龄区间    最多夺冠体重区间   最多夺冠身高区间

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 创建通用区间函数
def create_bin(column, bins, labels):
    bin_expr = F.when((column >= bins[0]) & (column < bins[1]), labels[0])
    for i in range(1, len(bins) - 1):
        bin_expr = bin_expr.when((column >= bins[i]) & (column < bins[i+1]), labels[i])
    return bin_expr

# 定义区间和标签
age_bins = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age_labels = ["10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80"]

height_bins = [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220]
height_labels = ["120-125", "125-130", "130-135", "135-140", "140-145", "145-150", "150-155", "155-160", "160-165", "165-170", "170-175", "175-180", "180-185", "185-190", "190-195", "195-200", "200-205", "205-210", "210-215", "215-220"]

weight_bins = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
weight_labels = ["30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "100-110", "110-120", "120-130", "130-140", "140-150"]

# 添加区间列
spark_df = spark_df.withColumn("Age_Bin", create_bin(F.col("Age"), age_bins, age_labels))
spark_df = spark_df.withColumn("Height_Bin", create_bin(F.col("Height"), height_bins, height_labels))
spark_df = spark_df.withColumn("Weight_Bin", create_bin(F.col("Weight"), weight_bins, weight_labels))

# 过滤出金牌选手
gold_medalists = spark_df.filter(spark_df["Medal"] == "Gold")

# 分组统计不同运动的最多夺冠年龄区间、身高区间和体重区间
age_mode = gold_medalists.groupBy("Sport", "Age_Bin").count().withColumnRenamed("count", "Age_Count")
height_mode = gold_medalists.groupBy("Sport", "Height_Bin").count().withColumnRenamed("count", "Height_Count")
weight_mode = gold_medalists.groupBy("Sport", "Weight_Bin").count().withColumnRenamed("count", "Weight_Count")

# 找到每个运动最多夺冠的年龄区间
age_mode = age_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Age_Count"))))
age_mode = age_mode.filter(age_mode["Row_Number"] == 1).drop("Row_Number")

# 找到每个运动最多夺冠的身高区间
height_mode = height_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Height_Count"))))
height_mode = height_mode.filter(height_mode["Row_Number"] == 1).drop("Row_Number")

# 找到每个运动最多夺冠的体重区间
weight_mode = weight_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Weight_Count"))))
weight_mode = weight_mode.filter(weight_mode["Row_Number"] == 1).drop("Row_Number")

# 合并结果
result = age_mode.join(height_mode, on="Sport").join(weight_mode, on="Sport")

# 选择需要的列并重命名
result = result.select("Sport", "Age_Bin", "Height_Bin", "Weight_Bin")

result.show()

result.explain(True)










# EDA4：Do host countries win significantly more medals compared to the Olympics before and after they host?
#比较主办国在主办和非主办时期的奖牌总数。，只对比主办的那一届往前两届和往后两届，然后新增一列，主办的那一届奖牌数比非主办的四届奥运奖牌数的最大值多百分之多少。



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, greatest

# 假设你已经创建了SparkSession，并导入了spark_df
# spark = SparkSession.builder.appName("Olympic Analysis").getOrCreate()

# 筛选出有奖牌的记录
medals_df = spark_df.filter(col("Medal") != 'NA')

# 初始化主办国家字典
host_countries = {
    1956: 'AUS', 1960: 'ITA', 1964: 'JPN', 1968: 'MEX', 1972: 'FRG', 1976: 'CAN',
    1980: 'URS', 1984: 'USA', 1988: 'KOR', 1992: 'ESP', 1996: 'USA', 2000: 'AUS',
    2004: 'GRE', 2008: 'CHN', 2012: 'GBR', 2016: 'BRA'
}

# 创建一个临时视图以便使用SQL查询
medals_df.createOrReplaceTempView("medals")

# 结果列表
results = []

# 计算每一届主办国的奖牌数和前后两届非主办时期的奖牌数
for year, country in host_countries.items():
    query = f"""
        SELECT
            {year} as Year,
            '{country}' as Country,
            COUNT(CASE WHEN NOC = '{country}' AND Year = {year} THEN Event END) as Medals_host,
            COUNT(CASE WHEN NOC = '{country}' AND Year = {year-4} THEN Event END) as Medals_nothost1,
            COUNT(CASE WHEN NOC = '{country}' AND Year = {year-8} THEN Event END) as Medals_nothost2,
            COUNT(CASE WHEN NOC = '{country}' AND Year = {year+4} THEN Event END) as Medals_nothost3,
            COUNT(CASE WHEN NOC = '{country}' AND Year = {year+8} THEN Event END) as Medals_nothost4
        FROM medals
    """
    result = spark.sql(query)
    results.append(result)

# 将所有结果合并到一个DataFrame中
final_df = results[0]
for df in results[1:]:
    final_df = final_df.union(df)

# 新增一列，计算主办时期的奖牌数比非主办时期的最大值多百分之多少
final_df = final_df.withColumn(
    "Medals_bigger%",
    (col("Medals_host") - greatest("Medals_nothost1", "Medals_nothost2", "Medals_nothost3", "Medals_nothost4"))
    / col("Medals_host") * 100
)

final_df.show()
