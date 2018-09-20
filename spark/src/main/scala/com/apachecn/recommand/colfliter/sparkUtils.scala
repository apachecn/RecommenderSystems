package com.apachecn.recommand.colfliter

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ych on 2018/9/20.
  */
object sparkUtils {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("utils").setMaster("local")
    val sc = new SparkContext(conf)

    def selectData(): Unit ={
      val in = "C:\\dtworkspace\\recommand\\data\\ratings"
      val rdd =sc.textFile(in).map(x=>(x,x.split("::")(1).toInt)).filter(x=>(x._2 < 1000)).map(_._1).coalesce(1).saveAsTextFile(in+"out")

    }
  }

}
