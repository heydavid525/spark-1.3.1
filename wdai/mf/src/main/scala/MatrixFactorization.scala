
import scala.collection.mutable

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

object MatrixFactorization extends App {

  case class Params(
      input: String = null,
      kryo: Boolean = false,
      numIterations: Int = 20,
      lambda: Double = 1.0,
      rank: Int = 10,
      numUserBlocks: Int = -1,
      numProductBlocks: Int = -1,
      useSimpleALS: Boolean = false) extends AbstractParams[Params]

  val defaultParams = Params()

  val parser = new OptionParser[Params]("MatrixFactorization") {
    head("MatrixFactorization: an example app for MatrixFactorization.")
    opt[Int]("rank")
      .text(s"rank, default: ${defaultParams.rank}}")
      .action((x, c) => c.copy(rank = x))
    opt[Int]("numIterations")
      .text(s"number of iterations, default: ${defaultParams.numIterations}")
      .action((x, c) => c.copy(numIterations = x))
    opt[Double]("lambda")
      .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
      .action((x, c) => c.copy(lambda = x))
    opt[Unit]("kryo")
      .text("use Kryo serialization")
      .action((_, c) => c.copy(kryo = true))
    opt[Int]("numUserBlocks")
      .text(s"number of user blocks, default: ${defaultParams.numUserBlocks} (auto)")
      .action((x, c) => c.copy(numUserBlocks = x))
    opt[Int]("numProductBlocks")
      .text(s"number of product blocks, " +
        s"default: ${defaultParams.numProductBlocks} (auto)")
      .action((x, c) => c.copy(numProductBlocks = x))
    opt[Unit]("useSimpleALS")
      .text("use simple ALS")
      .action((_, c) => c.copy(useSimpleALS = true))
    arg[String]("<input>")
      .required()
      .text("input paths to a MovieLens dataset of ratings")
      .action((x, c) => c.copy(input = x))
    note(
      """
        |For example, the following command runs this app on a synthetic dataset:
        |
        | bin/spark-submit --class org.apache.spark.examples.mllib.ALS \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  --rank 5 --numIterations 20 --lambda 1.0 --kryo \
        |  data/mllib/sample_movielens_data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    System.exit(1)
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"MatrixFactorizationALS with $params")
    /*
    if (params.kryo) {
      conf.registerKryoClasses(Array(classOf[mutable.BitSet], classOf[Rating]))
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    */
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val dataLoadingTimer = new Timer
    val data = sc.textFile(params.input)
    val training = data.map(_.split("\\s+") match { case Array(user, item, rate)
      => Rating(user.toInt, item.toInt, rate.toDouble) }).cache()

    val numTraining = training.count()
    val numUsers = training.map(_.user).distinct().count()
    val numMovies = training.map(_.product).distinct().count()

    println(s"Got $numTraining training ratings from $numUsers users on "
      + s"$numMovies movies.")
    val dataLoadingTime = dataLoadingTimer.elapsed
    println(s"Data loading time: $dataLoadingTime")

    val trainingTimer = new Timer
    val model = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setUserBlocks(params.numUserBlocks)
      .setProductBlocks(params.numProductBlocks)
      .run(training)
    val trainingTime = trainingTimer.elapsed
    val numIterations = params.numIterations
    println(s"Training time: $trainingTime ($numIterations Iterations)")

    val evalTimer = new Timer
    val (sqError, rmse) = computeErrors(model, training)
    val evalTime = trainingTimer.elapsed
    println(s"Eval time: $evalTime")

    println(s"Training SE = $sqError; RMSE = $rmse.")

    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeErrors(model: MatrixFactorizationModel, data: RDD[Rating]) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    val mse = predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean()
    val numData = data.count
    (mse * numData, math.sqrt(mse))
  }
}
