# pipeline_03_optimization.py
#
# Ce script mesure l'impact d'une optimisation clé : le Broadcast Join.
# 1. Lit la couche "silver".
# 2. Simule une jointure entre une grande table (faits) et une petite (dimension).
# 3. Mesure le temps d'exécution avec un Sort-Merge Join (shuffle).
# 4. Mesure le temps d'exécution avec un Broadcast Join.
# 5. Affiche les plans d'exécution pour prouver la différence.

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

def create_spark_session(app_name="Accidents Optimization"):
    """Crée et retourne une session Spark."""
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()

def chrono(label, action):
    """Exécute une action (fonction sans argument) et mesure le temps."""
    start_time = time.perf_counter()
    result = action()
    duration = time.perf_counter() - start_time
    print(f"[{label}] Temps d'exécution : {duration:.3f} s")
    return result

if __name__ == "__main__":
    spark = create_spark_session()

    # --- 1. Lecture des données Silver ---
    print("Reading silver data for join benchmark...")
    silver_path = "data/silver/accidents_2023_silver"
    df_silver = spark.read.parquet(silver_path)

    # Création d'une petite table de dimension (les départements)
    df_departements = df_silver.select("dep").distinct().withColumn("nom_region", F.lit("Region_Test"))
    print(f"Table de faits (df_silver): {df_silver.count()} lignes")
    print(f"Table de dimension (df_departements): {df_departements.count()} lignes")

    # --- 2. Benchmark de la jointure ---
    print("\n" + "="*50)
    print("BENCHMARK : SORT-MERGE JOIN vs BROADCAST JOIN")
    print("="*50)

    # Approche 1 : Sort-Merge Join (avec shuffle)
    # On désactive le broadcast automatique pour forcer le shuffle
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    
    def join_avec_shuffle():
        # Jointure classique qui va provoquer un shuffle
        return df_silver.join(df_departements, "dep").count()

    chrono("Sort-Merge Join (avec shuffle)", join_avec_shuffle)
    
    # Approche 2 : Broadcast Join
    # On réactive le broadcast (ou on le force manuellement)
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10m") # Valeur par défaut

    def join_avec_broadcast():
        # On indique explicitement à Spark de diffuser la petite table
        return df_silver.join(F.broadcast(df_departements), "dep").count()

    chrono("Broadcast Join (sans shuffle)", join_avec_broadcast)

    # --- 3. Analyse des plans d'exécution ---
    print("\n" + "="*50)
    print("PLANS D'EXÉCUTION PHYSIQUES")
    print("="*50)

    print("\n--- Plan pour le Sort-Merge Join (chercher 'SortMergeJoin' et 'Exchange') ---")
    df_silver.join(df_departements, "dep").explain()

    print("\n--- Plan pour le Broadcast Join (chercher 'BroadcastHashJoin' et l'absence d'Exchange) ---")
    df_silver.join(F.broadcast(df_departements), "dep").explain()

    spark.stop()