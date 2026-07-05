# pipeline_02_analysis.py
#
# Ce script réalise la deuxième étape du pipeline : l'analyse.
# 1. Lit les données nettoyées depuis la couche "silver" (Parquet).
# 2. Effectue une première analyse : corrélation entre la météo et la gravité des accidents.
# 3. Écrit le résultat agrégé dans la couche "gold".

import os
import sys

# Correction pour faire fonctionner Spark sur Windows
if sys.platform == "win32":
    hadoop_home = 'C:\\hadoop'
    os.environ['HADOOP_HOME'] = hadoop_home
    os.environ['PATH'] = f"{hadoop_home};{os.environ['PATH']}"

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def create_spark_session(app_name="Accidents Analysis"):
    """Crée et retourne une session Spark."""
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()

if __name__ == "__main__":
    spark = create_spark_session()

    # --- 1. Lecture de la couche Silver ---
    print("Reading silver data...")
    silver_path = "data/silver/accidents_2023_silver"
    # On définit des chemins de sortie distincts pour chaque analyse
    gold_path_meteo = "data/gold/analyse_meteo"
    gold_path_vehicule = "data/gold/analyse_vehicule"
    gold_path_top_dep = "data/gold/top3_departements_par_mois"

    df_silver = spark.read.parquet(silver_path)

    # Mettre le DataFrame en cache car il sera probablement réutilisé pour d'autres analyses
    df_silver.cache()
    df_silver.count() # Action pour matérialiser le cache

    # --- 2. Analyse 1 : Gravité des accidents selon la météo (agrégation) ---
    print("Computing analysis: accident gravity by weather conditions...")

    # On mappe les codes numériques de la météo (colonne 'atm') à des libellés clairs
    weather_desc = (
        F.when(F.col("atm") == 1, "Normale")
        .when(F.col("atm") == 2, "Pluie légère")
        .when(F.col("atm") == 3, "Pluie forte")
        .when(F.col("atm") == 4, "Neige - grêle")
        .when(F.col("atm") == 5, "Brouillard - fumée")
        .when(F.col("atm") == 6, "Vent fort - tempête")
        .when(F.col("atm") == 7, "Temps éblouissant")
        .when(F.col("atm") == 8, "Temps couvert")
        .otherwise("Autre/Non renseigné")
    )

    analyse_meteo = (
        df_silver
        .withColumn("meteo", weather_desc)
        .groupBy("meteo")
        .agg(
            F.count("*").alias("total_impliques"),
            F.sum(F.when(F.col("grav") == 2, 1).otherwise(0)).alias("tues"), # grav=2 signifie "Tué"
            F.sum(F.when(F.col("grav") == 3, 1).otherwise(0)).alias("blesses_hospitalises"), # grav=3
            F.round((F.sum(F.when(F.col("grav").isin(2, 3), 1).otherwise(0)) / F.count("*")) * 100, 2).alias("taux_gravite_pct")
        )
        .orderBy(F.desc("taux_gravite_pct"))
    )

    print("--- Analysis 1 Results (Weather) ---")
    analyse_meteo.show(truncate=False)

    # --- 3. Analyse 2 : Gravité par catégorie de véhicule (jointure implicite) ---
    print("\nComputing analysis: accident gravity by vehicle category...")
    
    vehicle_desc = (
        F.when(F.col("catv") == 1, "Bicyclette")
        .when(F.col("catv").isin(2, 30, 31, 32, 33, 34), "Deux-roues motorisé")
        .when(F.col("catv").isin(3, 7), "Voiture (VL)")
        .when(F.col("catv").isin(10, 13, 14, 15, 17), "Utilitaire / Poids Lourd")
        .when(F.col("catv").isin(37, 38, 39, 40), "Transports en commun")
        .otherwise("Autre / Inconnu")
    )

    analyse_vehicule = (
        df_silver
        .withColumn("categorie_vehicule", vehicle_desc)
        .groupBy("categorie_vehicule")
        .agg(
            F.count("*").alias("total_impliques"),
            F.sum(F.when(F.col("grav") == 2, 1).otherwise(0)).alias("tues"),
            F.sum(F.when(F.col("grav") == 3, 1).otherwise(0)).alias("blesses_hospitalises"),
            F.round((F.sum(F.when(F.col("grav").isin(2, 3), 1).otherwise(0)) / F.count("*")) * 100, 2).alias("taux_gravite_pct")
        )
        .orderBy(F.desc("taux_gravite_pct"))
    )
    
    print("--- Analysis 2 Results (Vehicle) ---")
    analyse_vehicule.show(truncate=False)

    # --- 4. Analyse 3 : Top 3 des départements par mois (window function) ---
    print("\nComputing analysis: Top 3 departments by accidents per month...")

    # On compte d'abord le nombre d'accidents distincts par mois et département
    accidents_par_dep_mois = (
        df_silver.select("Num_Acc", "mois", "dep")
        .distinct()
        .groupBy("mois", "dep")
        .agg(F.count("Num_Acc").alias("nb_accidents"))
    )

    # On définit la fenêtre pour classer les départements au sein de chaque mois
    fenetre_mois = Window.partitionBy("mois").orderBy(F.desc("nb_accidents"))

    top_departements = (
        accidents_par_dep_mois
        .withColumn("rang", F.row_number().over(fenetre_mois))
        .filter(F.col("rang") <= 3)
        .orderBy("mois", "rang")
    )

    print("--- Analysis 3 Results (Top Departments) ---")
    top_departements.show(36, truncate=False) # 12 mois * 3 = 36 lignes

    # --- 5. Écriture de la couche Gold ---
    print(f"\nWriting gold data...")
    analyse_meteo.coalesce(1).write.mode("overwrite").option("header", "true").csv(gold_path_meteo)
    analyse_vehicule.coalesce(1).write.mode("overwrite").option("header", "true").csv(gold_path_vehicule)
    top_departements.coalesce(1).write.mode("overwrite").option("header", "true").csv(gold_path_top_dep)

    df_silver.unpersist()
    spark.stop()