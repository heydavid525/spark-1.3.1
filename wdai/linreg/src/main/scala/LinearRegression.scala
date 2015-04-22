/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * An example app for linear regression. Run with
 *  A synthetic dataset can be found at `data/mllib/sample_linear_regression_data.txt`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object LinearRegression extends App {

  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  object DataFormat extends Enumeration {
    type DataFormat = Value
    val EntryList, LibSVM = Value
  }

  import RegType._
  import DataFormat._

  case class Params(
      input: String = null,
      numIterations: Int = 100,
      stepSize: Double = 1.0,
      regType: RegType = L2,
      regParam: Double = 0.01,
      dataFormat: DataFormat = EntryList) extends AbstractParams[Params]

  val defaultParams = Params()

  val parser = new OptionParser[Params]("LinearRegression") {
    head("LinearRegression: an example app for linear regression.")
    opt[Int]("numIterations")
      .text("number of iterations")
      .action((x, c) => c.copy(numIterations = x))
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
    arg[String]("<input>")
      .required()
      .text("input paths to data in (i) EntryList format, without .X or .Y "
        + "(2) LIBSVM format")
      .action((x, c) => c.copy(input = x))
    note(
      """
        |For example, the following command runs this app on a synthetic dataset:
        |
        |  bin/spark-submit --class org.apache.spark.examples.mllib.LinearRegression \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  data/mllib/sample_linear_regression_data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    sys.exit(1)
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LinearRegression with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)
    println(s"trainParams: $params")

    val dataLoadingTimer = new Timer
    val training = params.dataFormat match {
      case EntryList =>
        // Convert row and column indices from 1-based to 0-based.
        val Y = sc.textFile(params.input + ".Y")
          .map(_.trim)
          .filter(line => !(line.isEmpty || line.startsWith("%")))
          .map(_.split("\\s+") match {
            case Array(rowId, colId, value)
          => (rowId.toInt - 1, value.toDouble) })

        val parsedX = sc.textFile(params.input + ".X")
          .map(_.trim)
          .filter(line => !(line.isEmpty || line.startsWith("%")))
          .map(_.split("\\s+") match {
            case Array(rowId, colId, value)
            => (rowId.toInt - 1, (colId.toInt - 1, value.toDouble))
          }) .groupByKey() .map{case (rowId, cols) => {
            val (indices, values) = cols.toList.sortWith(_._1 < _._1).unzip
            (rowId, indices.toArray, values.toArray)}
          }

        // Determine feature dim.
        parsedX.persist(StorageLevel.MEMORY_ONLY)
        val dim = parsedX.map { case (rowId, indices, values) =>
          indices.lastOption.getOrElse(0)
        }.reduce(math.max) + 1

        val X = parsedX.map{case(rowId, indices, values) =>
            (rowId,  Vectors.sparse(dim, indices, values))}
        X.join(Y).map{case(rowId, (x, y)) => LabeledPoint(y, x)}.cache()

      case LibSVM => MLUtils.loadLibSVMFile(sc, params.input).cache()
    }
    val numTrain = training.count()
    val dataLoadingTime = dataLoadingTimer.elapsed
    println(s"Data loading time: $dataLoadingTime")

    println(s"numTrain: $numTrain")

    //examples.unpersist(blocking = false)

    val updater = params.regType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val trainingTimer = new Timer
    val algorithm = new LinearRegressionWithSGD()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setStepSize(params.stepSize)
      .setUpdater(updater)
      .setRegParam(params.regParam)

    val model = algorithm.run(training)
    val trainingTime = trainingTimer.elapsed
    val numIterations = params.numIterations
    println(s"Training time: $trainingTime ($numIterations Iterations)")

    val evalTimer = new Timer
    val prediction = model.predict(training.map(_.features))
    val predictionAndLabel = prediction.zip(training.map(_.label))
    val w = model.weights.toArray
    var nnz = 0.0
    for (i <- 0 to (w.length - 1)) {
      if (w(i) != 0) nnz += 1
    }
    println(s"nnz in w: $nnz")
    //println("learned w: " + w.slice(0,1000).mkString(" "))
    var l1 = 0.0;
    for (i <- 0 to (w.length - 1)) {
      l1 += math.abs(w(i))
    }

    val trainSqError = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)
    val mse = trainSqError / numTrain.toDouble
    val obj = 0.5 * trainSqError + params.regParam * l1

    val evalTime = evalTimer.elapsed
    println(s"Train SE = $trainSqError; Train MSE = $mse; " +
      s"Train objective: $obj; eval time: $evalTime")

    sc.stop()
  }
}
