package apache.wiki

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * @author ${user.name}
 */
object App {
  
  // def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)
  
  // def main(args : Array[String]) {
  //   println( "Hello World!" )
  //   println("concat arguments = " + foo(args))
  // }

  def main(args: Array[String]) {
    // 初始化 SparkContext对象，通过SparkConf指定配置的内容
    val conf = new SparkConf().setMaster("local").setAppName("My App")
    val sc = new SparkContext(conf)
    println("this system exit ok!!!")
    
    // 每一个 JVM 可能只能激活一个 SparkContext 对象。在创新一个新的对象之前，必须调用 stop() 该方法停止活跃的 SparkContext。
    sc.stop()
  }
}
