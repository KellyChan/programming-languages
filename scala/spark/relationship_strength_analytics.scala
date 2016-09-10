// $ spark-shell —jars /usr/lib/avro/avro-mapred.jar \
// —conf spark.serializer=org.apache.spark.serializer.KryoSerializer

// First we're going to import the classes we need and open some of the files
// we imported from our relational database into Hadoop with Sqoop

import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred.{AvroInputFormat, AvroWrapper}
import org.apache.hadoop.io.NullWritable

val warehouse = "hdfs://quickstart/user/hive/warehouse"

val order_items_path = warehouse + "order_items"
val order_items = sc.hadoopFile[AvroWrapper[GenericRecord], NullWritable, AvroInputFormat[GenericRecord]](order_items_path)

val products_path = warehouse + "products"
val products = sc.hadoopFile[AvroWrapper[GenericRecord], NullWritable, AvroInputFormat[GenericRecord]](product_path)



// Next, we extract the fields from order_items and products that we care about
// and get a list of every product, its name and quantity, grouped by order

val orders = order_items.map { x => (
  x._1.datum.get("order_item_product_id"),
  (x._1.datum.get("order_item_order_id"), x._1.datum.get("order_item_quantity"))
)}.join(
  products.map { x => (
    x._1.datum.get("product_id"),
    (x._1.datum.get("product_name"))
)}).map(x => (
  scala.Int.unbox(x._2._1._1), // order_id
  (
    scala.Int.unbox(x._2._1._2), // quantity
    x._2._2.toString // product_name
  )
)).groupByKey()


// Finally, we tally how many times each combination of products appears
// together in an order, and print the 10 most common combinations.

val cooccurrences = orders.map(order => (
  order._1,
  order._2.toList.combinations(2).map(order_pair => (
    if (order_pair(0)._2 < order_pair(1)._2) (order_pair(0)._2, order_pair(1)._2)
    else (order_pair(1)._2, order_pair(0)._2), order_pair(0)._1 * order_pair(1)._1
  ))
))

val combos = cooccurrrences.flatMap(x => x._2).reduceByKey((a,b) => a+b)
val mostCommon = combos.map(x => (x._2, x._1)).sortByKey(false).take(10)

println(mostCommon.deep.mkString("\n"))
