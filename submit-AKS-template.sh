#!bin/bash

spark-submit \
  --master k8s://https://###YOUR AKS API SERVER ADDRESS### \
  --deploy-mode cluster \
  --name ###PROGRAM NAME### \
  --conf spark.kubernetes.container.image=###Your SPARK IMAGE### \
  --conf spark.kubernetes.driver.pod.name=###PROGRAM NAME###\
  --conf spark.executor.cores=1 \
  --conf spark.executor.instances=2 \
  --conf spark.kubernetes.context=###YOUR AKS CLUSTER### \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=###SERVICE ACCOUNT### \
  --conf spark.hadoop.fs.azure.account.key.###STORAGE ACCOUNT###.dfs.core.windows.net=#########STORAGE ACCOUNT ACCESS KEY########################### \
  --jars local:///opt/spark/jars/hadoop-azure-3.3.4.jar,local:///opt/spark/jars/hadoop-azure-datalake-3.3.4.jar \
  --conf spark.kubernetes.file.upload.path=abfss://###BLOB CONTAINER###@###STORAGE ACCOUNT###.dfs.core.windows.net/ \
  --verbose \
  file:///home/spark/mount/###PROGRAM_PATH###


