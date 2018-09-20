package com.apachecn.recommand.colfliter

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.BitSet

/**
  * Created by ych on 2018/9/20.
  * 基于物品的协同过滤
  */
class ItemCF {
  /**
    * 使用BitSet 计算jaccard 距离
    */
  def computeJaccardSim(sc: SparkContext,
                        pathIn: String): RDD[(String,Double)] ={
    val rdd = sc.textFile(pathIn)
      .map(_.split(" ", 2)(1))
      .zipWithIndex()
      .map(x => (x._2.toInt,x._1.split(" ", -1)))
      .map(x=>{
        for (i <- x._2) yield {
          (i.split("\\:")(0), x._1)
        }
      }).flatMap(x=>x)
      .map(x=>(x._1,BitSet(x._2.toString.toInt))).reduceByKey(_.union(_))

    val re = rdd.cartesian(rdd).map {
      case((key0,set0),(key1,set1))=>{
        val key=key0+"|"+key1
        val j = (set0 &(set1)).size
        val q = set0.union(set1).size
        val re = j.toDouble/q
        (key, re)
      }
    }
    re
  }




}

object ItemCF{

}
