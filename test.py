import sys
import re
import numpy as np
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('read txt').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+


schema = types.StructType([ # commented-out fields won't be read
    types.StructField('r', types.IntegerType(), False),
    types.StructField('g', types.IntegerType(), False),
    types.StructField('b', types.IntegerType(), False),
])


def some_function(path):
    return (path[0:11])

def split_func(string):
    return [int(x) for x in string.split(",")]

path_to_hour = functions.udf(some_function,
                                returnType=types.StringType())

def main(in_directory, out_directory):
    ###
    sc = spark.sparkContext
    # Load a text file and convert each line to a Row.
    lines = sc.textFile("/home/hla115/sfuhome/cmpt318/project/wfile")
    parts = lines.map(split_func)
    #parts = lines.map(lambda l: int(x) for x in l.split(","))
    # Infer the schema, and register the DataFrame as a table.
    schemaPeople = spark.createDataFrame(parts,schema)
    schemaPeople.show()
    # SQL can be run over DataFrames that have been registered as a table.

    # The results of SQL queries are Dataframe objects.
    # rdd returns the content as an :class:`pyspark.RDD` of :class:`Row`.

    '''comments = spark.read.csv(in_directory,sep=' ', schema=schema).withColumn('filename', functions.input_file_name())
                filtered_comments = comments.filter(comments['language'] == 'en')
                filtered_comments = filtered_comments.filter(filtered_comments['title'] != "Main_Page")
                filtered_comments = filtered_comments.filter(~(filtered_comments.title.like("%Special:%")))
                match = re.compile(r'20[0-9]*-[0-9].')
                selected = filtered_comments.select(
                    filtered_comments['title'],
                    filtered_comments['requests'],
                    path_to_hour(functions.split(filtered_comments['filename'],'pagecounts-')[2]).alias('time'),
                    ).cache()
                grouped = selected.groupBy(selected['time'])
                groups = grouped.agg(
                    functions.max(selected['requests']).alias('max')
                    )
                selected = selected.join(groups,on="time")
                selected = selected.filter(selected['requests'] == selected["max"])
                selected = selected.select(
                    selected['time'],
                    selected['title'],
                    selected['requests'],
                    )
                selected = selected.sort(selected['time'])
            
                selected.write.csv(out_directory + '-problem2', mode='overwrite')'''
    ##selected.show()

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
