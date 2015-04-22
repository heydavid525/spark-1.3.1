
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
//import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.SparseVector
import scala.collection.mutable.ListBuffer
import java.io._
import scala.math.Ordering
import scala.io.Source

//import grizzled.slf4j.Logger
import grizzled.slf4j.Logging

object LogisticRegression extends App with Logging {
  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  object DataFormat extends Enumeration {
    type DataFormat = Value
    val LibSVM = Value
  }

  /*
  object PointOrdering extends Ordering[LabeledPoint] {
    def compare(a:LabeledPoint, b:LabeledPoint) = {
      val a1 = a.features.asInstanceOf[SparseVector].values(0)
      val b1 = a.features.asInstanceOf[SparseVector].values(0)
      a1 compare b1
    }
  }
  */

  import RegType._
  import DataFormat._

  case class Params(
      input: String = null,
      numIterations: Int = 100,
      stepSize: Double = 1.0,
      regType: RegType = L2,
      regParam: Double = 0.01,
      minibatchFraction: Double = 1,
      dataFormat: DataFormat = LibSVM,
      numLegs: Int = 1,
      numDups: Int = 1,
      numParallelism: Int = 1) extends AbstractParams[Params]

  val defaultParams = Params()

  val parser = new OptionParser[Params]("LogisticRegression") {
    head("LogisticRegression: an example app for linear regression.")
    opt[Int]("numIterations")
      .text("number of iterations")
      .action((x, c) => c.copy(numIterations = x))
    opt[Int]("numDups")
      .text("number of times to duplicated the data")
      .action((x, c) => c.copy(numDups = x))
    opt[Double]("stepSize")
      .text(s"initial step size, default: ${defaultParams.stepSize}")
      .action((x, c) => c.copy(stepSize = x))
    opt[String]("regType")
      .text(s"regularization type (${RegType.values.mkString(",")}), " +
      s"default: ${defaultParams.regType}")
      .action((x, c) => c.copy(regType = RegType.withName(x)))
    opt[String]("dataFormat")
      .text(s"data format (${DataFormat.values.mkString(",")}), " +
      s"default: ${defaultParams.dataFormat}")
      .action((x, c) => c.copy(dataFormat = DataFormat.withName(x)))
    opt[Double]("regParam")
      .text(s"regularization parameter, default: ${defaultParams.regParam}")
      .action((x, c) => c.copy(regParam = x))
    opt[Double]("minibatchFraction")
      .text(s"fraction of points to use per epoch: ${defaultParams.minibatchFraction}")
    opt[Int]("numLegs")
      .text("number of legs to ge to numIterations: ${defaultParams.numLegs}")
      .action((x, c) => c.copy(numLegs = x))
    opt[Int]("numParallelism")
      .text("number of partitions")
      .action((x, c) => c.copy(numParallelism = x))
    arg[String]("<input>")
      .required()
      .text("input paths to data in (i) EntryList format, without .X or .Y "
        + "(2) LIBSVM format")
      .action((x, c) => c.copy(input = x))
    note(
      """
        |For example, the following command runs this app on a synthetic dataset:
        |
        |  bin/spark-submit --class org.apache.spark.examples.mllib.LogisticRegression \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  data/mllib/sample_linear_regression_data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    sys.exit(1)
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def parseLine(line: String, dim: Integer) = {
    val items = line.split(' ')
    val label = items.head.toDouble
    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      //val index = indexAndValue(0).toInt - 1 // Convert 1-based indices
      //to 0-based.
      val index = indexAndValue(0).toInt
      val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip
    LabeledPoint(label, Vectors.sparse(dim, indices.toArray, values.toArray))
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"LogisticRegression with $params")
      //.set("spark.executor.memory", "125g")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)
    //println(s"Experiment params: $params")
    println(s"Experiment params: $params")

    val dataLoadingTimer = new Timer
    val dirName = "/l0/criteo/"
    val dim = 118468612   // criteo 3 days.
    val idx = sc.parallelize(1 to params.numParallelism, params.numParallelism)
    val numSlaves = 48
    val idxCount = idx.count()
    println(s"idxCount: $idxCount, numSlaves: $numSlaves")
    val fileString = idx.mapPartitions {
      iter =>
        val idx = iter.next()
        if (idx % numSlaves == 0) {
          val dirNameLocal = dirName
          val f = new File(dirNameLocal)
          val files = f.listFiles
          if (files != null) {
            val listBuff = new ListBuffer[String]()
            for (filename <- files) {
              listBuff += filename + s"|$idx"
            }
            listBuff.toList.iterator
          } else {
            val listBuff = new ListBuffer[String]()
            listBuff += s"should not happen:$idx"
            listBuff.toList.iterator
          }
        } else {
          val listBuff = new ListBuffer[String]()
          listBuff += s"noop:$idx"
          listBuff.toList.iterator
        }
    }
    val files = fileString.collect.mkString(",")
    val gotNumFiles = fileString.count
    println(s"Got $gotNumFiles files $files")

    val training = idx.mapPartitions {
      iter =>
        //assert iter.hasNext()
        val idx = iter.next()
        if (idx % numSlaves == 0) {
          val dirNameLocal = dirName
          val f = new File(dirNameLocal)
          val files = f.listFiles
          if (files != null) {
            val listBuff = new ListBuffer[LabeledPoint]()
            for (filename <- files) {
              val br = new BufferedReader(new FileReader(filename))
              try {
                var line = br.readLine()
                while (line != null) {
                  listBuff += parseLine(line, dim)
                  line = br.readLine()
                }
              } finally {
                br.close
              }
/*
              for (line <- Source.fromFile(filename).getLines()) {
                listBuff += parseLine(line, dim)
              }
*/
              //listBuff += file + iter.mkString(",")
            }
            listBuff.toList.iterator
            //files.map(_.getName).toList.iterator
          } else {
            val listBuff = new ListBuffer[LabeledPoint]()
            listBuff.toList.iterator
          }
      } else {
        val listBuff = new ListBuffer[LabeledPoint]()
        listBuff.toList.iterator
      }
    }.cache()
    //val files = localFiles.collect.mkString(",")
    //val gotNumFiles = localFiles.count
    //println(s"Got $gotNumFiles files $files")
    //val numTrain = training.count
    //println(s"Got $numTrain training")
/*
    val training = sc.parallelize((1 to numSlaves), numSlaves).flatMap {
      i =>
        val listBuf = new ListBuffer[LabeledPoint]
        if (i % 16 == 0) {
          info(s"i = $i")
          val files = new File(dirName).listFiles
          info(s"files $files")
          for (filename <- files) {
            //println(s"Reading $filename ...")
            info(s"Reading $filename ...")
            for (line <- Source.fromFile(filename).getLines()) {
              listBuf += parseLine(line, dim)
            }
          }
        }
        listBuf.toList
    }
*/
    //val numFeatures = 501
/*
    val training = if (params.numDups == 1) {
      //MLUtils.loadLibSVMFile(sc, params.input).cache()
      MLUtils.loadLibSVMFile(sc, params.input, numFeatures, params.numParallelism).cache()
    } else {
      val training1x = MLUtils.loadLibSVMFile(sc, params.input).cache()
      val numDups = params.numDups
      def dupData(p: LabeledPoint) : List[LabeledPoint] = {
        val listBuffer = new ListBuffer[LabeledPoint]
        val dim = p.features.size
        val sparseVec = p.features.asInstanceOf[SparseVector]
        for (i <- 0 to (numDups - 1)) {
          listBuffer += new LabeledPoint(p.label, Vectors.sparse(dim,
            sparseVec.indices, sparseVec.values))
        }
        listBuffer.toList
      }
      val trainingToReturn = training1x.flatMap{p => dupData(p)}.cache()
      val numTrain1x = training1x.count()
      println(s"Duplicate $numTrain1x data by factor: $numDups")
      training1x.unpersist()
      trainingToReturn
    }
*/
    val numTrain = training.count()
    val dataLoadingTime = dataLoadingTimer.elapsed
    println(s"Data loading time: $dataLoadingTime")

    println(s"numTrain: $numTrain")
    val updater = params.regType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val trainingTimer = new Timer
    val numItersPerLeg = math.ceil(params.numIterations.toDouble /
          params.numLegs)
    val featureDim = training.first().features.size
    val training2 = training.repartition(params.numParallelism).cache()
    training.unpersist()
    var currWeights = Vectors.dense(new Array[Double](featureDim))
    val numLegs = params.numLegs - 1
    for (iLeg <- 0 to (params.numLegs-1)) {
      println(s"Leg: $iLeg/$numLegs")
      val beginIter = numItersPerLeg * iLeg
      val endIter = math.min(beginIter + numItersPerLeg,
        params.numIterations).toInt
      val numIterThisLeg = (endIter - beginIter).toInt
      println(s"numIterThisLeg: $numIterThisLeg")
      val algorithm = new LogisticRegressionWithSGD()
      algorithm.optimizer
        .setNumIterations(numIterThisLeg)
        .setStepSize(params.stepSize / math.sqrt(beginIter + 1))
        .setUpdater(updater)
        .setRegParam(params.regParam)
        .setMiniBatchFraction(params.minibatchFraction)
      val model = algorithm.run(training2, currWeights)
      val trainingTime = trainingTimer.elapsed
      println(s"Training time: $trainingTime (reaching $endIter iterations)")

      val trainErrorTimer = new Timer
      val prediction = model.predict(training2.map(_.features))
      val predictionAndLabel = prediction.zip(training2.map(_.label))
      val trainError = predictionAndLabel.map { case (p, l) =>
        if (p == l) 0
        else 1
        //math.abs(p - l)
      }.reduce(_ + _) / numTrain.toDouble
      val trainErrorTime = trainErrorTimer.elapsed
      println(s"Train error: $trainError (eval time: $trainErrorTime)")

      val trainObjTimer = new Timer
      val w_array = model.weights.toArray
      //val outputFile = "/tank/projects/biglearning/wdai/spark/myspark/wdai/output/lr.weight"
      //printToFile(new File(outputFile))
      //  { p =>
      //    w_array.foreach(p.println)
      //  }
      //println(s"Saved weight to $outputFile")
      currWeights = Vectors.dense(w_array)
      val w_brz = new BDV[Double](model.weights.toArray)
      val bias = model.intercept
      val regObj = params.regType match {
        case NONE => 0
        case L1 =>
          var l1 = 0.0
          for (i <- 0 to (w_array.length - 1)) {
            l1 += math.abs(w_array(i))
          }
          params.regParam * l1
        case L2 =>
          var l2 = 0.0
          for (i <- 0 to (w_array.length - 1)) {
            l2 += w_array(i) * w_array(i)
          }
          0.5 * params.regParam * l2  // 1/2 * lambda * ||w||^2
      }

      val localWeights = w_brz
      val bcWeights = training2.context.broadcast(localWeights)

/*
      val dirName = "/l0/criteo/"
      val dim = 118468612   // criteo 3 days.
      val numSlaves = 3 * 16
      val localFiles = training.mapPartitions {
        iter =>
          val dirNameLocal = dirName
          val f = new File(dirNameLocal)
          val files = f.listFiles
          if (files != null) {
            files.map(_.getName).toList.iterator
          } else {
            val listBuff = new ListBuffer[String]()
            listBuff += "null"
            listBuff.toList.iterator
          }
      }
      val files = localFiles.collect.mkString(",")
      val numFiles = localFiles.count()
      println(s"Got $numFiles files $files")
*/

      val logisticLoss = training2.mapPartitions { iter =>
        val bias_local = bias
        val w_brz_local = bcWeights.value
        iter.map { labeledPoint =>
          val feature = labeledPoint.features.asInstanceOf[SparseVector]
          val feature_brz = new BSV[Double](feature.indices, feature.values,
            feature.size)
          val dotProd = w_brz_local.dot(feature_brz) + bias_local
          labeledPoint.label match {
            case 0 => math.log(1 + math.exp(dotProd))
            case 1 => math.log(1 + math.exp(-dotProd))
          }
        }
      }.reduce(_+_) / numTrain.toDouble
      val objValue = logisticLoss + regObj
      val trainObjTime = trainObjTimer.elapsed
      println(s"Logistic Loss: $logisticLoss; Train obj: "
        + s"$objValue (iter $endIter); Eval time: $trainObjTime")
    }

    sc.stop()
  }

}
