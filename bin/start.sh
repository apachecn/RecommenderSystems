#!bin/bash

mvn clean
mvn compile
mvn package
if [ $? -eq 0 ]
then
    echo "打包成功"
else
    echo "打包失败"
    exit 1
fi
# /opt/spark-2.0.0-bin-hadoop2.6/bin/spark-submit --class apache.wiki.App /opt/git/RecommenderSystems/target/RecommenderSystems-1.0-SNAPSHOT.jar
# /opt/spark-2.0.0-bin-hadoop2.6/bin/spark-submit --class apache.wiki.WordCount /opt/git/RecommenderSystems/target/RecommenderSystems-1.0-SNAPSHOT.jar
# /opt/spark/bin/spark-submit --class apache.wiki.OfflineRecommender /opt/git/RecommenderSystems/target/RecommenderSystems-1.0-SNAPSHOT.jar
# /opt/spark/bin/spark-submit --class apache.wiki.StreamingWordCount /opt/git/RecommenderSystems/target/RecommenderSystems-1.0-SNAPSHOT.jar
/opt/spark/bin/spark-submit --class apache.wiki.OnlineRecommender /opt/git/RecommenderSystems/target/RecommenderSystems-1.0-SNAPSHOT.jar
echo 'spark-submit success'
