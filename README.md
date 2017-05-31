name := "NetflixRecommender"
 
version := "1.0"
  
scalaVersion := "2.10.4"
libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.4.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.4.0"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.10" % "1.4.0"
libraryDependencies += "org.apache.spark" % "spark-streaming-kafka_2.10" % "1.4.0"
libraryDependencies += "org.mongodb" %% "casbah" % "3.0.0"
libraryDependencies += "org.jblas" % "jblas" % "1.2.4"

mergeStrategy in assembly <<= (mergeStrategy in assembly) { mergeStrategy => {
    case entry => {
      val strategy = mergeStrategy(entry)
      if (strategy == MergeStrategy.deduplicate) MergeStrategy.first
      else strategy
    }
  }
}
