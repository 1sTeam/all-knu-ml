FROM ubuntu:20.04

# set a directory for the app
WORKDIR /app

# copy all the files to the container
COPY . .

#install jdk
RUN apt-get update -y
RUN apt-get install -y openjdk-11-jdk

RUN apt-get install -y python3-pip

RUN apt-get install -y tzdata

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# run the command
CMD ["python3", "kafka_python.py"]
