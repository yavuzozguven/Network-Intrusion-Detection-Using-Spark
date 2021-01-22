package com.yavuzozguven

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

class Predict(val test : DataFrame) {
  def main(): Unit ={

    var t0 = System.currentTimeMillis()
    val LRmodel = LogisticRegressionModel.load("lrmodel.model")
    val lr_preds = LRmodel.transform(test)
    var t1 = System.currentTimeMillis()
    val lr_ms = t1 - t0


    t0 = System.currentTimeMillis()
    val DTmodel = DecisionTreeClassificationModel.load("dtmodel.model")
    val dt_preds = DTmodel.transform(test)
    t1 = System.currentTimeMillis()
    val dt_ms = t1 - t0


    t0 = System.currentTimeMillis()
    val RFmodel = RandomForestClassificationModel.load("rfmodel.model")
    val rf_preds = RFmodel.transform(test)
    t1 = System.currentTimeMillis()
    val rf_ms = t1 - t0

    println(s"Logistic regression in $lr_ms ms")
    //lr_preds.select("label","prediction").show()
    println(s"Decision Tree in $dt_ms ms")
    //dt_preds.select("label","prediction").show()
    println(s"Random Forest in $rf_ms ms")
    //rf_preds.select("label","prediction").show()


    /*Write results to CSV*/
    val lr_df = lr_preds.select("label","prediction")
    val dt_df = dt_preds.select("label","prediction")
    val rf_df = rf_preds.select("label","prediction")
    lr_df.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("outlr")
    dt_df.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("outdt")
    rf_df.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("outrf")
  }
}
