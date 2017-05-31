#!bin/bash

mvn clean
mvn compile
mvn package
echo 'package success'
# /opt/spark-2.0.0-bin-hadoop2.6/bin/spark-submit --class apache.wiki.App /opt/git/RecommendedSystem/target/RecommendedSystem-1.0-SNAPSHOT.jar
# /opt/spark-2.0.0-bin-hadoop2.6/bin/spark-submit --class apache.wiki.WordCount /opt/git/RecommendedSystem/target/RecommendedSystem-1.0-SNAPSHOT.jar
/opt/spark/bin/spark-submit --class apache.wiki.OfflineRecommender /opt/git/RecommendedSystem/target/RecommendedSystem-1.0-SNAPSHOT.jar
echo 'spark-submit success'
