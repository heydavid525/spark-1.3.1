
import scala.collection.mutable

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.SimpleALS
import org.apache.spark.ml.{Rating => R}
import scala.collection.mutable.ListBuffer

object MatrixFactorization extends App {

  case class Params(
      input: String = null,
      kryo: Boolean = false,
      numIterations: Int = 20,
      lambda: Double = 1.0,
      rank: Int = 10,
      numUserBlocks: Int = -1,
      numProductBlocks: Int = -1,
      dupUsers: Int = 1,
      dupMovies: Int = 1) extends AbstractParams[Params]

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
    opt[Int]("dupUsers")
      .text(s"duplicate users if dupUsers > 1, default: ${defaultParams.dupUsers}")
      .action((x, c) => c.copy(dupUsers = x))
    opt[Int]("dupMovies")
      .text(s"duplicate movies if dupMovies > 1, default: ${defaultParams.dupMovies}")
      .action((x, c) => c.copy(dupMovies = x))
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
    val training1x = data.map(_.split("\\s+") match { case Array(user, item, rate)
      => new R(user.toInt, item.toInt, rate.toFloat) }).cache()

    val numTraining1x = training1x.count()
    val numUsers1x = training1x.map(_.user).distinct().count()
    val numMovies1x = training1x.map(_.product).distinct().count()
    // +1 as user id starts from 0, so interval needs to avoid overlap.
    val maxUsers1x = training1x.map(_.user).distinct().max() + 1
    val maxMovies1x = training1x.map(_.product).distinct().max() + 1

    println(s"param setting: $params")
    println(s"Duplicating $numTraining1x training ratings from $numUsers1x users on "
      + s"$numMovies1x movies. maxUsers1x: $maxUsers1x; maxMovie1x: $maxMovies1x")

    // Duplicate along users.
    val dupUsers = params.dupUsers
    def dupUser(r: R) : List[R] = {
      val listBuffer = new ListBuffer[R]
      for (i <- 0 to (dupUsers - 1)) {
        val dupUser = r.user + maxUsers1x * i
        listBuffer += new R(dupUser.toInt, r.product, r.rating)
      }
      listBuffer.toList
    }

    val dupMovies = params.dupMovies
    def dupProduct(r: R) : List[R] = {
      val listBuffer = new ListBuffer[R]
      for (i <- 0 to (dupMovies - 1)) {
        val dupMovie = r.product + maxMovies1x * i
        listBuffer += new R(r.user, dupMovie.toInt, r.rating)
      }
      listBuffer.toList
    }

    val training = training1x.flatMap{ r => dupUser(r) }.cache()
      .flatMap { r => dupProduct(r) }.cache()
    training1x.unpersist()

    val numTraining = training.count()
    val numUsers = training.map(_.user).distinct().count()
    val numMovies = training.map(_.product).distinct().count()

    println(s"After dup: numTraining: $numTraining; numUsers: $numUsers;"
      + s" numMovies: $numMovies")


    val dataLoadingTime = dataLoadingTimer.elapsed
    println(s"Data loading time: $dataLoadingTime")

    val trainingTimer = new Timer
    val numBlocks = 10
    val (userFactors, prodFactors) =
      new SimpleALS().run(training
      , params.rank
      , numBlocks
      , params.numIterations
      , params.lambda
      , false  // no implicit pref
      , 1.0)    // ignored when no implicit pref
    val uf = userFactors.map{
      case (id, arr) => (id, arr.map(_.toDouble))}.cache()
    val pf = prodFactors.map{
      case (id, arr) => (id, arr.map(_.toDouble))}.cache()
    val model = new MatrixFactorizationModel(params.rank, uf, pf)
    val trainingTime = trainingTimer.elapsed
    val numIterations = params.numIterations
    println(s"Training time: $trainingTime ($numIterations Iterations)")

    val evalTimer = new Timer
    val trainingRating = training.map(x => Rating(x.user, x.product, x.rating))
    val (sqError, rmse) = computeErrors(model, trainingRating)
    val evalTime = evalTimer.elapsed
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
