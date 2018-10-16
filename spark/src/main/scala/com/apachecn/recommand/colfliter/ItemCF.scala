package com.apachecn.recommand.colfliter

import breeze.linalg.SparseVector
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, SparseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.collection.BitSet

/**
  * Created by ych on 2018/9/20.
  * 基于物品的协同过滤
  */
class ItemCF {

  def computeJaccardSimWithDF(sc: SparkContext,
                                  featurePath: String
                             ): CoordinateMatrix ={
    val sqlContext = new SQLContext(sc)
    val rdd = sqlContext.read.parquet(featurePath).select("features")
      .rdd.map(x=>x(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])
      .map(x=>(x.indices))
      .zipWithIndex()
      .map(x=>{
        for (i <- x._1) yield {
          (x._2, i)
        }
      })
      .flatMap(x=>x)
      .map(x=>(x._1,BitSet(x._2.toString.toInt)))
      .reduceByKey(_.union(_))

    val entries = rdd.cartesian(rdd).map {
      case ((key0, set0), (key1, set1)) => {
        val j = (set0 & (set1)).size
        val q = set0.union(set1).size
        val re = j.toDouble / q
        MatrixEntry(key0.toInt, key1.toInt, re)
      }
    }
    val simMat: CoordinateMatrix = new CoordinateMatrix(entries)
    simMat
  }


  def computeJaccardSimWithLibSVM(sc: SparkContext,
                        featurePath: String): CoordinateMatrix ={
    val rdd = sc.textFile(featurePath)
      .map(_.split(" ", 2)(1))
      .zipWithIndex()
      .map(x => (x._2.toInt,x._1.split(" ", -1)))
      .map(x=>{
        for (i <- x._2) yield {
          (i.split("\\:")(0), x._1)
        }
      }).flatMap(x=>x)
      .map(x=>(x._1,BitSet(x._2.toString.toInt)))
      .reduceByKey(_.union(_))

    val entries = rdd.cartesian(rdd).map {
      case((key0,set0),(key1,set1))=>{
        val j = (set0 &(set1)).size
        val q = set0.union(set1).size
        val re = j.toDouble/q
        MatrixEntry(key0.toInt,key1.toInt,re)
      }
    }
    val simMat: CoordinateMatrix = new CoordinateMatrix(entries)
    simMat
  }


  def computeCosSimWithDF(sc: SparkContext,
                              featurePath: String):  CoordinateMatrix ={

    val sqlContext = new SQLContext(sc)
    val rdd = sqlContext.read.parquet(featurePath).select("features")
      .rdd.map(x=>x(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
    val mat = new RowMatrix(rdd)
    val simMat = mat.columnSimilarities()
    simMat
  }

  /**
    *
    * @param sc
    * @param featureSize 特征数量
    * @param featurePath
    */
  def computeCosSimWithLibSVM(sc: SparkContext,
                    featureSize: Int,
                    featurePath: String):  CoordinateMatrix ={

    val rows = sc.textFile(featurePath).map(_.split(" "))
      .map(x=>(x.filter(g=>g.contains(":"))))
      .map(x=>(x.map(_.split(":")).map(ar => (ar(0).toInt,ar(1).toDouble))))
      .map(x=>(Vectors.sparse(featureSize,x)))

    val mat = new RowMatrix(rows)
    val simMat = mat.columnSimilarities()
    simMat
  }


  def loadSimMatrix(sc: SparkContext,
                    simPath: String,
                    featruesSize: Int
                   ):  SparseMatrix ={
    val entries = sc.textFile(simPath)
      .map(_.split("\\|", -1))
      .map(x=>(x(0).toInt, x(1).toInt, x(2).toDouble))
      .collect()
    val simMatrix = SparseMatrix.fromCOO(featruesSize, featruesSize, entries)
    simMatrix
  }

  /**
    * 将相似矩阵存储为文本文件
    * @param sc
    * @param savePath
    * @param mat
    */
  def saveSimMatrix(sc: SparkContext,
                    savePath: String,
                    mat: CoordinateMatrix): Unit ={
    val sim = mat
    sim.entries.map(x=>x.i+"|"+x.j+"|"+x.value).coalesce(1).saveAsTextFile(savePath)

  }

  def predictByMatrix(sc: SparkContext,
              simMatrix: breeze.linalg.Matrix[Double],
              featuresSize: Int,
              featurePath: String,
              resultPath: String
             ): Unit ={
    val rdd = sc.textFile(featurePath)
      .map(_.split(" "))
      .map(x=>(x.filter(g=>g.contains(":"))))
      .map(x=>(x.map(_.split(":")).map(ar => (ar(0).toInt,ar(1).toDouble))))
      .map(x=>{
        val idx = x.map(_._1)
        val v = x.map(_._2)
        val vec: SparseVector[Double] = new SparseVector(idx, v, featuresSize)
        vec
      })
//      .map(x=>(x.toDenseVector.toDenseMatrix.dot(simMatrix)))
  }


}

object ItemCF extends ItemCF{
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("utils").setMaster("local[8]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val libsvmFeaturePath = "C:\\workspace\\data\\apacheCN\\libsvmOut"
    val dfFeaturePath = "C:\\workspace\\data\\apacheCN\\dfOut"
    val simPath = "C:\\workspace\\data\\apacheCN\\simPath"

//    val JaccardSimPath = "..//data//jaccardSim"
    val CosSimPath = "..//data//cosSim"
    val featureSize = 3953

//
//    val sim1 = computeJaccardSimWithLibSVM(sc,libsvmFeaturePath)
//    sim1.entries.take(10)

    val sim2 = computeCosSimWithLibSVM(sc,featureSize,libsvmFeaturePath)
//    sim2.entries.take(10).foreach(println)
//    saveSimMatrix(sc,simPath,sim2)

        val sim3 = computeJaccardSimWithDF(sc,dfFeaturePath)
    sim3.entries.take(10)

    val sim4 = computeCosSimWithDF(sc,dfFeaturePath)
    sim4.entries.take(10)

//    computeItemJaccardSim(sc,featurePath, JaccardSimPath)
//    computeItemCosSim(sc,100,featurePath, CosSimPath)
    val simMatrix = loadSimMatrix(sc, simPath, featureSize)
//    val score = predict()
  }
}
