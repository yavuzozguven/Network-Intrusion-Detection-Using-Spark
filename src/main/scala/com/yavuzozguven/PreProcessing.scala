package com.yavuzozguven

import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id

class PreProcessing(var df : DataFrame) {

  def main(): DataFrame ={
    /*Select features*/
    var df_num = df.select("state","Dttl","synack"
      ,"swin","dwin","ct_state_ttl","ct_src_ltm"
      ,"ct_srv_dst","Sttl","ct_dst_sport_ltm","Djit")

    /*String indexing on state feature*/
    val strIndexer = new StringIndexer()
      .setInputCol("state")
      .setOutputCol("indexedState")
      .fit(df_num)

    df_num = strIndexer.transform(df_num)
    df_num = df_num.drop("state")


    /*Change numerical columns types*/
    val num_categories = df_num.dtypes.filter (_._2 == "StringType") map (_._1)

    num_categories.foreach{r=>
      df_num = df_num.withColumn(r,df_num(r).cast("double"))
    }

    /*Select label column*/
    var df_label = df.select("label")

    /*Change type of label column*/
    df_label = df_label.withColumn("label",df_label("label").cast("integer"))

    /*Merge 2 dataframe*/
    val df1 = df_num.withColumn("row_id", monotonically_increasing_id())

    val df2 = df_label.withColumn("row_id", monotonically_increasing_id())

    var df_final = df1.join(df2, ("row_id")).drop("row_id")
    df_num.drop("row_id")
    df_label.drop("row_id")

    /*Vectorization*/
    val assembler = new VectorAssembler()
      .setInputCols(df_num.columns)
      .setOutputCol("features")

    df_final = assembler.transform(df_final)

    /*Scaling*/
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)
      .fit(df_final)

    df_final = scaler.transform(df_final)

    return df_final
  }
}
