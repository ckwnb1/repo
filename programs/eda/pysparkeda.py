from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count, when, avg, lag, format_number, concat, lit, row_number, concat_ws, collect_list, expr,greatest
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql import functions as F

# Create SparkSession
spark = SparkSession.builder \
    .appName("Athlete Events Data Import") \
    .getOrCreate()

# Define schema
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

# File path
csv_file_path = "/home/spark/mount/eda/athlete_events.csv"
output_base_path = "/home/spark/mount/output"

# Read CSV file
spark_df = spark.read.csv(csv_file_path, schema=schema, header=True)
# Filter data
spark_df = spark_df.filter((spark_df.Games.contains('Summer')) & (spark_df.Year >= 1956))

# Print schema
spark_df.printSchema()

# Show first 5 rows
spark_df.show(5)

# Get statistical information of numeric columns
spark_df.describe().show()

# EDA1: Distribution and development trends of sports participation between male and female athletes
# 1. Calculate the number of male and female athletes participating in each Olympic Games 
# 2. Calculate the growth rate and ratio of female participants in each Olympic Games
#  Group by gender and year, and count the number of athletes in each group
gender_trend_df = spark_df.groupBy("Sex", "Year").agg(count("ID").alias("Participant_Count"))
#  Sort the result
sorted_gender_trend_df = gender_trend_df.orderBy("Year", "Sex")
# Show the first 21 rows
sorted_gender_trend_df.show(20)
#  Group by gender and year, and count the number of athletes in each group
gender_trend_df = spark_df.groupBy("Sex", "Year").agg(
    count("ID").alias("Participant_Count")
)
# Calculate annual growth rate and participation ratio
window_spec = Window.partitionBy("Sex").orderBy("Year")
gender_trend_df = gender_trend_df.withColumn(
    "Prev_Year_Participant", lag("Participant_Count").over(window_spec)
).withColumn(
    "Growth_Rate", 
    ((col("Participant_Count") - col("Prev_Year_Participant")) / col("Prev_Year_Participant") * 100).cast("decimal(10,3)")
).withColumn(
    "Growth_Rate", concat(col("Growth_Rate"), lit("%"))
)
#Calculate the total number of participants and participation ratio each year
total_participants_df = spark_df.groupBy("Year").agg(count("ID").alias("Total_Participants"))
gender_ratio_df = gender_trend_df.join(total_participants_df, on="Year").withColumn(
    "Participation_Ratio", (col("Participant_Count") / col("Total_Participants") * 100).cast("decimal(10,3)")
).withColumn(
    "Participation_Ratio", concat(col("Participation_Ratio"), lit("%"))
)
# Capture physical plan
# Show all data
gender_ratio_df.orderBy("Year", "Sex").show(n=200000, truncate=False)
gender_ratio_df.orderBy("Year", "Sex").write.csv(f"{output_base_path}/eda1_gender_ratio.csv", header=True, mode="overwrite")







# EDA2: Identification of the sports infrastructure and traditional strong events of various countries
# Idea: Track the change in the total number of medals over time for each country 
# 2. Filter the leading countries in each sport and finally merge them. 
# For example, if France is the leading country in both sport1 and sport2, 
# the result will show France: sport1, sport2
medal_counts = spark_df.groupBy("Year", "NOC").agg(count(when(col("Medal").isNotNull(), 1)).alias("MedalCount"))
# Use window function to find the top 10 countries with the most medals each year
windowSpec = Window.partitionBy("Year").orderBy(col("MedalCount").desc())
top_10_countries = medal_counts.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") <= 10).drop("rank")
# Display the results as the top 10 countries with the most medals and the corresponding number of medals each year
result = top_10_countries.groupBy("Year").agg(concat_ws(", ", collect_list(concat_ws("-", col("NOC"), col("MedalCount")))).alias("Top10Countries"))
# Show results
result.show(20, truncate=False)
sport_leaders = spark_df.groupBy("NOC", "Sport").agg(count(when(col("Medal").isNotNull(), 1)).alias("MedalCount"))
# Find the leading country in each sport
windowSpec = Window.partitionBy("Sport").orderBy(col("MedalCount").desc())
sport_leaders = sport_leaders.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") == 1).drop("rank")
# Merge all leading projects of the same country
merged_sport_leaders = sport_leaders.groupBy("NOC").agg(concat_ws(", ", collect_list(concat_ws("-", col("Sport"), col("MedalCount")))).alias("DominantSports"))
# Show results
merged_sport_leaders.show(40, truncate=False)
result.write.csv(f"{output_base_path}/eda2_top10_countries.csv", header=True, mode="overwrite")
merged_sport_leaders.write.csv(f"{output_base_path}/eda2_dominant_sports.csv", header=True, mode="overwrite")



# EDA3: The Relationship Between Height Vs Weight Vs Age of Participants Across Sports
# Analyze the distribution of winning age, height, and weight in various sports. 
# In other words, divide the age into intervals from 10 to 80, every 5 years, 10-15, 15-20.....,
# and the height and weight into different intervals.
# The final result is similar to: Sport1, the most winning age interval, the most winning weight interval, the most winning height interval

# Create a general bin function
def create_bin(column, bins, labels):
    bin_expr = F.when((column >= bins[0]) & (column < bins[1]), labels[0])
    for i in range(1, len(bins) - 1):
        bin_expr = bin_expr.when((column >= bins[i]) & (column < bins[i+1]), labels[i])
    return bin_expr

# Define bins and labels
age_bins = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age_labels = ["10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80"]

height_bins = [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220]
height_labels = ["120-125", "125-130", "130-135", "135-140", "140-145", "145-150", "150-155", "155-160", "160-165", "165-170", "170-175", "175-180", "180-185", "185-190", "190-195", "195-200", "200-205", "205-210", "210-215", "215-220"]

weight_bins = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
weight_labels = ["30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "100-110", "110-120", "120-130", "130-140", "140-150"]

# Add bin columns
spark_df = spark_df.withColumn("Age_Bin", create_bin(F.col("Age"), age_bins, age_labels))
spark_df = spark_df.withColumn("Height_Bin", create_bin(F.col("Height"), height_bins, height_labels))
spark_df = spark_df.withColumn("Weight_Bin", create_bin(F.col("Weight"), weight_bins, weight_labels))

# Filter out gold medalists
gold_medalists = spark_df.filter(spark_df["Medal"] == "Gold")

# Group statistics for the most winning age interval, height interval, and weight interval in different sports
age_mode = gold_medalists.groupBy("Sport", "Age_Bin").count().withColumnRenamed("count", "Age_Count")
height_mode = gold_medalists.groupBy("Sport", "Height_Bin").count().withColumnRenamed("count", "Height_Count")
weight_mode = gold_medalists.groupBy("Sport", "Weight_Bin").count().withColumnRenamed("count", "Weight_Count")

# Find the most winning age interval for each sport
age_mode = age_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Age_Count"))))
age_mode = age_mode.filter(age_mode["Row_Number"] == 1).drop("Row_Number")

# Find the most winning height interval for each sport
height_mode = height_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Height_Count"))))
height_mode = height_mode.filter(height_mode["Row_Number"] == 1).drop("Row_Number")

# Find the most winning weight interval for each sport
weight_mode = weight_mode.withColumn("Row_Number", F.row_number().over(Window.partitionBy("Sport").orderBy(F.desc("Weight_Count"))))
weight_mode = weight_mode.filter(weight_mode["Row_Number"] == 1).drop("Row_Number")

# Merge results
result = age_mode.join(height_mode, on="Sport").join(weight_mode, on="Sport")

# Select and rename required columns
result = result.select("Sport", "Age_Bin", "Height_Bin", "Weight_Bin")
result.show()
# Write EDA3 results to file
result.write.csv(f"{output_base_path}/eda3_height_weight_age.csv", header=True)




# EDA4: Do host countries win significantly more medals compared to the Olympics before and after they host?
# Compare the total number of medals won by host countries during the host period and non-host periods. 
# Only compare the host period with the two previous and two subsequent Olympics. 
# Add a new column showing the percentage increase in medals won during the host period compared to the maximum number of medals won during the four non-host periods.
# Assume you have created SparkSession and imported spark_df
# spark = SparkSession.builder.appName("Olympic Analysis").getOrCreate()

# Filter records with medals
medals_df = spark_df.filter(col("Medal") != 'NA')

# Initialize host countries dictionary
host_countries = {
    1956: 'AUS', 1960: 'ITA', 1964: 'JPN', 1968: 'MEX', 1972: 'FRG', 1976: 'CAN',
    1980: 'URS', 1984: 'USA', 1988: 'KOR', 1992: 'ESP', 1996: 'USA', 2000: 'AUS',
    2004: 'GRE', 2008: 'CHN', 2012: 'GBR', 2016: 'BRA'
}

# Create a temporary view for SQL queries
medals_df.createOrReplaceTempView("medals")

# Result list
results = []

# Calculate the number of medals won by the host country during the host period and the two previous and two subsequent Olympics
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

# Combine all results into one DataFrame
final_df = results[0]
for df in results[1:]:
    final_df = final_df.union(df)

# Add a new column to calculate the percentage increase in medals won during the host period compared to the maximum number of medals won during the non-host periods
final_df = final_df.withColumn(
    "Medals_bigger%",
    (col("Medals_host") - greatest("Medals_nothost1", "Medals_nothost2", "Medals_nothost3", "Medals_nothost4"))
    / col("Medals_host") * 100
)

final_df.show()
# Write EDA4 results to file
final_df.write.csv(f"{output_base_path}/eda4_host_country_medals.csv", header=True)