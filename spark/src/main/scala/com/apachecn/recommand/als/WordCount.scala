package apache.wiki

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * @author ${user.name}
 */
object WordCount {
    def main(args: Array[String]) {

        // 初始化 SparkContext对象，通过SparkConf指定配置的内容
        val conf = new SparkConf().setMaster("local").setAppName("My app") //.set("spark.executor.memory", "2g")
        val sc = new SparkContext(conf)

        // // 检验输入参数
        // if (args.length < 1) {
        //     println("USAGE:")
        //     println("spark-submit ... xxx.jar Date_String [Iteration]")
        //     println("spark-submit ... xxx.jar 20160424 10")
        //     sys.exit()
        // }

        val lines = sc.textFile("file:/opt/git/RecommenderSystems/README.md")
        lines.flatMap(_.split(" "))
            .map((_, 1))
            .reduceByKey(_+_)
            .map(x => (x._2, x._1))
            .sortByKey(false)
            .map(x => (x._2, x._1))
            .saveAsTextFile("file:/opt/git/RecommenderSystems/output/result.log")

        // println("this system exit ok!!!")

        // 每一个 JVM 可能只能激活一个 SparkContext 对象。在创新一个新的对象之前，必须调用 stop() 该方法停止活跃的 SparkContext。
        sc.stop()
    }

}
