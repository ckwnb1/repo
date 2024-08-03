# This is the basic Spark image for running a Spark environment
# on local machine. This setting does not support deploying the 
# Spark programs to AKS

FROM apache/spark

#promote privilege to root 
USER root

# Set basic runtime
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install vim -y
RUN apt-get install sudo -y
RUN pip3 install --upgrade pip setuptools

#For data analysis
RUN pip3 install numpy
RUN pip3 install pandas pyarrow

################################################################

#Set the home account for user spark
RUN mkdir /home/spark && chown spark:spark /home/spark

#Set up the env variables for Spark and 
ENV SPARK_HOME /opt/spark/
ENV PATH /opt/spark/bin:/opt/spark/sbin:$PATH

#Set the output log level to WARN
COPY ./log4j2.properties /opt/spark/conf/
#This is for the mac machine
RUN chmod 0644 /opt/spark/conf/log4j2.properties

################################################################

#Try to start the history server when starting
COPY start.sh /start.sh
RUN chmod 0755 /start.sh

#For spark history server to store log files
RUN mkdir /tmp/spark-events && chown spark:spark /tmp/spark-events

################################################################

#This is for jupyter lab
RUN apt install texlive-xetex texlive-fonts-recommended texlive-plain-generic -y

#go back to user spark account
USER spark
WORKDIR /home/spark

#Install Jupyter Lab and findspack package
RUN pip3 install jupyterlab findspark
ENV PATH /home/spark/.local/bin:$PATH

#set jupyter server
RUN jupyter server --generate-config
RUN echo "c.ServerApp.ip = '0.0.0.0'" > .jupyter/jupyter_server_config.py

################################################################

#4040 for spark UI, 18080 for spark history server, 8888 for Jupyter lab
EXPOSE 4040-4050
EXPOSE 18080
EXPOSE 8888

ENTRYPOINT ["/start.sh"]



