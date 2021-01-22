package com.yavuzozguven

import org.apache.spark.sql._
import org.apache.spark.sql.types._


object Main {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")
    val spark = SparkSession.builder.master("local").appName("Project").getOrCreate

    /*CSV Schema*/
    val schema = retSchema()

    /*Read CSV*/
    val df_train = spark.read.options(Map("sep"->",", "header"-> "true")).
      schema(schema).
      csv("unswtra.csv")
    val df_test = spark.read.options(Map("sep"->",", "header"-> "true")).
      schema(schema).
      csv("unswtest.csv")


    val training = new PreProcessing(df_train).main()
    val test = new PreProcessing(df_test).main()


    new Train(training).main()
    new Predict(test).main()
  }

  def retSchema(): StructType ={
     new StructType().
      add("id",StringType,true).
      add("dur",StringType,true).
      add("proto",StringType,true).
      add("service",StringType,true).
      add("state",StringType,true).
      add("spkts",StringType,true).
      add("dpkts",StringType,true).
      add("sbytes",StringType,true).
      add("dbytes",StringType,true).
      add("rate",StringType,true).
      add("sttl",StringType,true).
      add("dttl",StringType,true).
      add("sload",StringType,true).
      add("dload",StringType,true).
      add("sloss",StringType,true).
      add("dloss",StringType,true).
      add("sinpkt",StringType,true).
      add("dinpkt",StringType,true).
      add("sjit",StringType,true).
      add("djit",StringType,true).
      add("swin",StringType,true).
      add("stcpb",StringType,true).
      add("dtcpb",StringType,true).
      add("dwin",StringType,true).
      add("tcprtt",StringType,true).
      add("synack",StringType,true).
      add("ackdat",StringType,true).
      add("smean",StringType,true).
      add("dmean",StringType,true).
      add("trans_depth",StringType,true).
      add("response_body_len",StringType,true).
      add("ct_srv_src",StringType,true).
      add("ct_state_ttl",StringType,true).
      add("ct_dst_ltm",StringType,true).
      add("ct_src_dport_ltm",StringType,true).
      add("ct_dst_sport_ltm",StringType,true).
      add("ct_dst_src_ltm",StringType,true).
      add("is_ftp_login",StringType,true).
      add("ct_ftp_cmd",StringType,true).
      add("ct_flw_http_mthd",StringType,true).
      add("ct_src_ltm",StringType,true).
      add("ct_srv_dst",StringType,true).
      add("is_sm_ips_ports",StringType,true).
      add("attack_cat",StringType,true).
      add("label",StringType,true)
  }
}
