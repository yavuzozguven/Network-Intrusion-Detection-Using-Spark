package com.yavuzozguven

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.sql.DataFrame

class Train(val training : DataFrame) {
  def main(): Unit ={

    var t0 = System.currentTimeMillis()
    /*Logistic Regression*/
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.4)

    val model = lr.fit(training)
    model.save("lrmodel.model")
    var t1 = System.currentTimeMillis()
    println(s"Logistic regression in ${t1-t0} ms")


    t0 = System.currentTimeMillis()
    /*Decision Tree Classifier*/
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(30)

    val model_dt = dt.fit(training)
    model_dt.save("dtmodel.model")
    t1 = System.currentTimeMillis()
    println(s"Decision Tree in ${t1-t0} ms")

    t0 = System.currentTimeMillis()
    /*Random Forest Classifier*/
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)
      .setMaxDepth(30)

    val model_rf = rf.fit(training)
    model_rf.save("rfmodel.model")
    t1 = System.currentTimeMillis()
    println(s"Random Forest in ${t1-t0} ms")
  }
}
