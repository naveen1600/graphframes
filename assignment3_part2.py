import os
import urllib.request
import gzip
from pyspark.sql import SparkSession
from graphframes import GraphFrame

# DOWNLOAD AND EXTRACT DATASET
dataset_url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
gz_path = "wiki-Vote.txt.gz"
txt_path = "wiki-Vote.txt"

if not os.path.exists(txt_path):
    urllib.request.urlretrieve(dataset_url, gz_path)

    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())


spark = SparkSession.builder \
    .appName("GraphFramesAssignment") \
    .getOrCreate()

sc = spark.sparkContext

sc.setCheckpointDir("/tmp/graphframes-checkpoint")

# LOAD DATA INTO RDD
raw_rdd = sc.textFile(txt_path)

# Remove comments
edges_rdd = raw_rdd.filter(lambda line: not line.startswith("#"))

# Parse into (src, dst)
edges_parsed = edges_rdd.map(lambda line: line.split())

# Convert to DataFrame
edges_df = edges_parsed.toDF(["src", "dst"])

# Create vertices from unique IDs
vertices_df = (
    edges_parsed
    .flatMap(lambda x: x)
    .distinct()
    .map(lambda x: (x,))
    .toDF(["id"])
)

# CREATE GRAPHFRAME
g = GraphFrame(vertices_df, edges_df)

# Output folder
os.makedirs("output", exist_ok=True)

# TOP 5 OUTDEGREE
outdeg = g.outDegrees.orderBy("outDegree", ascending=False).limit(5)
outdeg.write.mode("overwrite").csv("output/outdegree")

# TOP 5 INDEGREE
indeg = g.inDegrees.orderBy("inDegree", ascending=False).limit(5)
indeg.write.mode("overwrite").csv("output/indegree")

# PAGERANK
pr = g.pageRank(resetProbability=0.15, maxIter=10)
top_pr = pr.vertices.orderBy("pagerank", ascending=False).limit(5)
top_pr.write.mode("overwrite").csv("output/pagerank")

# CONNECTED COMPONENTS
cc = g.connectedComponents()
top_cc = (
    cc.groupBy("component")
    .count()
    .orderBy("count", ascending=False)
    .limit(5)
)
top_cc.write.mode("overwrite").csv("output/components")

# TRIANGLE COUNT
from pyspark.storagelevel import StorageLevel

tri = g.triangleCount(storage_level=StorageLevel.MEMORY_AND_DISK)
top_tri = tri.select("id", "count").orderBy("count", ascending=False).limit(5)
top_tri.write.mode("overwrite").csv("output/triangles")

spark.stop()

