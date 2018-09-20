package apache.wiki

import org.apache.spark._
import org.apache.spark.streaming._

/**
 * @author ${user.name}
 * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 * 
 * 参考地址
 * 推荐系统: http://www.kuqin.com/shuoit/20151202/349305.html
 * ALS说明: http://www.csdn.net/article/2015-05-07/2824641
 */

object StreamingWordCount{

    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setMaster("local[2]").setAppName("StreamingWordCount")
        val ssc = new StreamingContext(conf, Seconds(1))

        // 创建一个将要连接到 hostname:port 的离散流，如 localhost:9999 
        val lines = ssc.socketTextStream("localhost", 9999) 
        
        // 将每一行拆分成单词 val words = lines.flatMap(_.split(" "))
        val words = lines.flatMap(_.split(" "))

        val pairs = words.map(word => (word, 1)) 
        val wordCounts = pairs.reduceByKey(_ + _) 
        
        // 在控制台打印出在这个离散流（DStream）中生成的每个 RDD 的前十个元素
        // 注意 : 必需要触发 action（很多初学者会忘记触发action操作，导致报错：No output operations registered, so nothing to execute） 
        wordCounts.print()

        ssc.start() // 启动计算 
        ssc.awaitTermination() // 等待计算的终止
    }
}