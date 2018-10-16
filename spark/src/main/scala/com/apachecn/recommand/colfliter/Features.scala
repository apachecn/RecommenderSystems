package com.apachecn.recommand.colfliter

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ych on 2018/9/20.
  */
class Features {


  /**
    * 将Rating 为 id features 形式的DataFrame
    */
  def changeRating2DF(sc: SparkContext,
                       ratingsPath: String,
                      size: Int,
                       OutPath: String
                     ): Unit ={
    val sqlContext = new SQLContext(sc)
    val ratingsRdd = sc.textFile(ratingsPath)
      .map(_.split("::"))
      .map(x=>(x(0),Array((x(1).toInt,x(2).toDouble))))
      .reduceByKey(_ ++ _)
      .map(x=>(x._1,Vectors.sparse(size,x._2)))

    val df = sqlContext.createDataFrame(ratingsRdd).toDF("id", "features")
        .write.parquet(OutPath)
  }

  /**
    * 将输入的打分值，转为libsvm
    * 例如： 输入为
    *   1::661::3::978302109
    *   1::914::3::978301968
    * 转化之后结果为
    *  1 661:3 914:3
    */
  def changeRatings2LibSVM(sc: SparkContext,
                            ratingsPath: String,
                           ratingsLibSVMPath: String): Unit ={
    val ratingsRdd = sc.textFile(ratingsPath)
      .map(_.split("::"))
      .map(x=>(x(0),Array((x(1).toInt,x(2).toInt))))
      .reduceByKey(_ ++ _)
      .map(x=>(x._1+" " + x._2.sortBy(_._1).map(x=>(f"${x._1}:${x._2}")).mkString(" ")))
      .saveAsTextFile(ratingsLibSVMPath)
  }
}

object Features{
  def main(args: Array[String]): Unit = {

    /**
      * 提交命令 spark-submit --master yarn-cluster <ratingsPath> <libSVMOutPath> <dfOutPath> <featureSize>
     */
    val conf = new SparkConf().setAppName("Features Prepare")
    val sc = new SparkContext(conf)
    val ratingsPath = args(0)
    val libSVMOutPath =  args(1)
    val dfOutPath =  args(2)
    val featureSize = args(3).toInt

    val testF = new Features

    testF.changeRatings2LibSVM(sc, ratingsPath, libSVMOutPath)
    testF.changeRating2DF(sc, ratingsPath, featureSize, dfOutPath)

  }
}
