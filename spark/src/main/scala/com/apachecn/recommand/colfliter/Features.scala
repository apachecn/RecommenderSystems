package com.apachecn.recommand.colfliter

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ych on 2018/9/20.
  */
object Features {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Features Prepare").setMaster("local")
    val sc = new SparkContext(conf)
    val ratingsPath = "C:\\dtworkspace\\recommand\\data\\ratings"
    val ratingsLibSVMPath =  "C:\\dtworkspace\\recommand\\data\\ratingslibsvm"

    /**
      * 将输入的打分值，转为稀疏矩阵
      * 例如： 输入为
      *   1::661::3::978302109
      *   1::914::3::978301968
      * 转化之后结果为
      *  1 661:3 914:3
      */
    def changeRatings2LibSVM(): Unit ={
       val ratingsRdd = sc.textFile(ratingsPath)
        .map(_.split("::"))
        .map(x=>(x(0),Array((x(1).toInt,x(2).toInt))))
        .reduceByKey(_ ++ _)
        .map(x=>(x._1+" " + x._2.sortBy(_._1).map(x=>(f"${x._1}:${x._2}")).mkString(" ")))
        .coalesce(1).saveAsTextFile(ratingsLibSVMPath)


    }

  }

}
