
# pipeline_01_ingestion.py
#
# Ce script réalise la première étape du pipeline ETL :
# 1. Définit les schémas explicites pour les 4 fichiers CSV sur les accidents de 2023.
# 2. Lit les données brutes (couche "bronze").
# 3. Effectue les jointures pour créer une table large.
# 4. Nettoie et transforme certaines colonnes (coordonnées GPS).
# 5. Écrit le résultat en Parquet (couche "silver"), partitionné par département.

import os
import sys

# Correction pour faire fonctionner Spark sur Windows
if sys.platform == "win32":
    # Le dossier C:\hadoop doit exister et contenir winutils.exe
    hadoop_home = 'C:\\hadoop'
    os.environ['HADOOP_HOME'] = hadoop_home
    # On ajoute le dossier des binaires Hadoop au PATH
    os.environ['PATH'] = f"{hadoop_home};{os.environ['PATH']}"

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType, DoubleType, TimestampType
import pyspark.sql.functions as F

def create_spark_session(app_name="Accidents Ingestion"):
    """Crée et retourne une session Spark."""
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()

# --- Définition des Schémas ---
# Basé sur l'inférence de schema_inspector.py

schema_caracteristiques = StructType([
    StructField("Num_Acc", StringType(), True), # Num_Acc est un identifiant, pas un nombre à calculer
    StructField("jour", IntegerType(), True),
    StructField("mois", IntegerType(), True),
    StructField("an", IntegerType(), True),
    StructField("hrmn", StringType(), True), # Lu comme texte, sera transformé en timestamp
    StructField("lum", IntegerType(), True),
    StructField("dep", StringType(), True),
    StructField("com", StringType(), True),
    StructField("agg", IntegerType(), True),
    StructField("int", IntegerType(), True),
    StructField("atm", IntegerType(), True),
    StructField("col", IntegerType(), True),
    StructField("adr", StringType(), True),
    StructField("lat", StringType(), True), # Lu comme texte à cause de la virgule
    StructField("long", StringType(), True) # Lu comme texte à cause de la virgule
])

schema_lieux = StructType([
    StructField("Num_Acc", StringType(), True),
    StructField("catr", IntegerType(), True),
    StructField("voie", StringType(), True),
    StructField("v1", StringType(), True),
    StructField("v2", StringType(), True),
    StructField("circ", IntegerType(), True),
    StructField("nbv", IntegerType(), True),
    StructField("vosp", IntegerType(), True),
    StructField("prof", IntegerType(), True),
    StructField("pr", StringType(), True),
    StructField("pr1", StringType(), True),
    StructField("plan", IntegerType(), True),
    StructField("lartpc", StringType(), True),
    StructField("larrout", StringType(), True),
    StructField("surf", IntegerType(), True),
    StructField("infra", IntegerType(), True),
    StructField("situ", IntegerType(), True),
    StructField("vma", IntegerType(), True)
])

schema_usagers = StructType([
    StructField("Num_Acc", StringType(), True),
    StructField("id_usager", StringType(), True),
    StructField("id_vehicule", StringType(), True),
    StructField("num_veh", StringType(), True),
    StructField("place", IntegerType(), True),
    StructField("catu", IntegerType(), True),
    StructField("grav", IntegerType(), True),
    StructField("sexe", IntegerType(), True),
    StructField("an_nais", IntegerType(), True),
    StructField("trajet", IntegerType(), True),
    StructField("secu1", IntegerType(), True),
    StructField("secu2", IntegerType(), True),
    StructField("secu3", IntegerType(), True),
    StructField("locp", IntegerType(), True),
    StructField("actp", StringType(), True),
    StructField("etatp", IntegerType(), True)
])

schema_vehicules = StructType([
    StructField("Num_Acc", StringType(), True),
    StructField("id_vehicule", StringType(), True),
    StructField("num_veh", StringType(), True),
    StructField("senc", IntegerType(), True),
    StructField("catv", IntegerType(), True),
    StructField("obs", IntegerType(), True),
    StructField("obsm", IntegerType(), True),
    StructField("choc", IntegerType(), True),
    StructField("manv", IntegerType(), True),
    StructField("motor", IntegerType(), True),
    StructField("occutc", IntegerType(), True)
])


if __name__ == "__main__":
    spark = create_spark_session()

    base_path = "data/datasets/accidents_2023"
    output_path = "data/silver/accidents_2023_silver"

    # --- 1. Lecture des 4 fichiers CSV (Bronze) ---
    print("Reading CSV files...")
    # Utilisation de l'option `encoding` pour s'assurer de la bonne lecture des caractères
    read_options = {"header": True, "sep": ";", "encoding": "UTF-8"}
    df_caracteristiques = spark.read.csv(f"{base_path}/caracteristiques-2023.csv", schema=schema_caracteristiques, **read_options)
    df_lieux = spark.read.csv(f"{base_path}/lieux-2023.csv", schema=schema_lieux, **read_options)
    df_usagers = spark.read.csv(f"{base_path}/usagers-2023.csv", schema=schema_usagers, **read_options)
    df_vehicules = spark.read.csv(f"{base_path}/vehicules-2023.csv", schema=schema_vehicules, **read_options)

    # --- 2. Jointure des DataFrames ---
    # On modélise la relation : un accident a des lieux et des véhicules,
    # et un véhicule a des usagers.
    print("Joining dataframes...")

    df_silver = (df_caracteristiques
                 .join(df_lieux, "Num_Acc", "left_outer")
                 .join(df_vehicules, "Num_Acc", "left_outer")
                 # On joint les usagers sur la clé composite pour lier un usager à son véhicule dans l'accident.
                 .join(df_usagers, ["Num_Acc", "id_vehicule", "num_veh"], "left_outer")
    )

    # --- 3. Nettoyage et Transformation ---
    print("Cleaning and transforming data...")
    df_silver = (df_silver
                 .withColumn("latitude", F.regexp_replace(F.col("lat"), ",", ".").cast(DoubleType()))
                 .withColumn("longitude", F.regexp_replace(F.col("long"), ",", ".").cast(DoubleType()))
                 .withColumn("timestamp_accident", F.to_timestamp(F.concat_ws("-", F.col("an"), F.col("mois"), F.col("jour"), F.col("hrmn")), "yyyy-M-d-H:mm"))
                 .drop("lat", "long")
    )

    # On supprime les lignes sans ID d'accident et les doublons sur la clé primaire (accident, vehicule, usager)
    df_silver = df_silver.filter(F.col("Num_Acc").isNotNull()).dropDuplicates(["Num_Acc", "id_vehicule", "id_usager"])

    # --- 4. Écriture de la couche Silver ---
    print(f"Writing silver data to {output_path}...")
    (df_silver.write
        .mode("overwrite")
        .partitionBy("dep") # Partitionner par département est une bonne pratique
        .parquet(output_path)
    )

    print("--- Ingestion complete! ---")

    # On vérifie le résultat
    print(f"Silver dataframe has {df_silver.count()} rows.")
    df_silver.printSchema()

    spark.stop()
