# pipeline_04_exploration.py
#
# Ce script réalise l'étape "Exploration" du projet.
# Il compare les performances d'une transformation simple implémentée de deux manières :
# 1. Avec une User-Defined Function (UDF) en Python standard.
# 2. Avec les fonctions natives de Spark SQL.
#
# L'objectif est de mesurer le coût de la sérialisation/désérialisation
# entre la JVM de Spark et l'interpréteur Python.

import os
import sys
import time

# Correction pour faire fonctionner Spark sur Windows
if sys.platform == "win32":
    hadoop_home = 'C:\\hadoop'
    os.environ['HADOOP_HOME'] = hadoop_home
    os.environ['PATH'] = f"{hadoop_home};{os.environ['PATH']}"

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

def create_spark_session(app_name="Accidents Exploration"):
    """Crée et retourne une session Spark."""
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()

def chrono(label, action):
    """Exécute une action (fonction sans argument) et mesure le temps."""
    start_time = time.perf_counter()
    result = action()
    duration = time.perf_counter() - start_time
    print(f"[{label}] Temps d'exécution : {duration:.3f} s")
    return result

@F.udf(returnType=StringType())
def get_day_period_udf(hour):
    """UDF Python pour catégoriser le moment de la journée."""
    if hour is None:
        return "Inconnu"
    if 6 <= hour < 12:
        return "Matin"
    elif 12 <= hour < 18:
        return "Après-midi"
    elif 18 <= hour < 22:
        return "Soir"
    else:
        return "Nuit"

if __name__ == "__main__":
    spark = create_spark_session()

    print("Reading silver data for UDF benchmark...")
    silver_path = "data/silver/accidents_2023_silver"
    df_silver = spark.read.parquet(silver_path).withColumn("hour", F.hour("timestamp_accident"))
    df_silver.cache().count() # Mettre en cache pour que la lecture ne biaise pas la mesure

    print("\n" + "="*50)
    print("BENCHMARK : UDF vs FONCTION NATIVE")
    print("="*50)

    # Approche 1 : UDF Python
    def transform_avec_udf():
        return df_silver.withColumn("periode_jour", get_day_period_udf(F.col("hour"))).count()

    chrono("Transformation avec UDF Python", transform_avec_udf)

    # Approche 2 : Fonctions natives Spark
    native_expr = (
        F.when((F.col("hour") >= 6) & (F.col("hour") < 12), "Matin")
        .when((F.col("hour") >= 12) & (F.col("hour") < 18), "Après-midi")
        .when((F.col("hour") >= 18) & (F.col("hour") < 22), "Soir")
        .otherwise("Nuit")
    )
    def transform_avec_native():
        return df_silver.withColumn("periode_jour", native_expr).count()

    chrono("Transformation avec fonctions natives", transform_avec_native)

    print("\nConclusion : Les fonctions natives sont beaucoup plus rapides car elles s'exécutent entièrement dans la JVM, évitant le coût de communication avec Python.")

    df_silver.unpersist()
    spark.stop()