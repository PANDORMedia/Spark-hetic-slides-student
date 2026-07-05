```markdown
# Rapport de projet - Pipeline Spark (ONISR 2023)

- **Équipe** : Théophane Kengni Sokoudjou
- **Jeu de données** : Accidents corporels de la circulation routière - ONISR 2023
- **Date** : 05 Juillet 2026

---

## 1. Jeu de données et schéma cible

- **Source et volume** : Le projet repose sur le millésime 2023 des données ouvertes publiées par l'ONISR (Sécurité Routière France). Le volume initial (couche Bronze) se décompose en 4 fichiers relationnels interconnectés par la clé d'accident unique `Num_Acc` :
  * `caracteristiques-2023.csv` : **54 822** accidents
  * `usagers-2023.csv` : **125 789** usagers impliqués
  * `vehicules-2023.csv`
  * `lieux-2023.csv`
  
  Le volume final consolidé, nettoyé et dédoublonné (couche Silver) atteint exactement **125 829** lignes, où chaque enregistrement représente un usager unique impliqué dans un accident.

- **Schéma cible** : Nous avons forcé l'abandon de l'inférence automatique (`inferSchema`) au profit d'un `StructType` explicite. `Num_Acc` et `dep` ont été typés en `StringType` pour conserver l'intégrité des zéros initiaux et des indicatifs spécifiques (ex: départements de la Corse ou de l'Outre-mer). Les variables catégorielles (`atm`, `lum`, `grav`, `catv`) ont été configurées en `IntegerType`. Enfin, les champs géospatiaux textuels (`lat`, `long`) ont été transformés et castés en variables numériques de type `DoubleType` (`latitude` et `longitude`).

- **Questions métier visées** :
  1. **Impact Environnemental** : Existe-t-il une corrélation mesurable entre les conditions météorologiques (`atm`) et la gravité/létalité individuelle d'un sinistre ?
  2. **Vulnérabilité Typologique** : Quel est le taux de gravité par type de moyen de transport (Deux-roues, Bicyclettes, VL, Poids Lourds, Transports en commun) ?
  3. **Concentration Territoriale** : Quels sont les trois départements français qui concentrent structurellement le plus grand nombre d'accidents mois par mois au cours de l'année ?

---

## 2. Pipeline (bronze -> silver -> gold)

```text
brut (bronze, 4 CSV) ➔ nettoyé (silver, Parquet unifié) ➔ agrégé (gold, 3 requêtes cibles)

```

* **Nettoyage appliqué** :
* **Nettoyage des chaînes de caractères** : Utilisation de `F.regexp_replace` pour purger les espaces insécables masqués (caractères blancs de séparation) au sein des colonnes d'identifiants critiques comme `id_vehicule` et `id_usager`.
* **Correction géospatiale** : Remplacement des virgules par des points dans les chaînes de coordonnées avant le transtypage en `DoubleType`.
* **Filtrage des valeurs aberrantes** : Isolation des années de naissance cohérentes (`an_nais` entre 1900 et 2024) et exclusion des codes de gravité non spécifiés (`grav != -1`).
* **Gestion de l'unicité** : Dédoublonnage via `dropDuplicates(["Num_Acc", "id_vehicule", "id_usager"])` pour obtenir le grain le plus fin sans sur-comptage lors des jointures.


* **Statistiques des volumes** :
* **Lignes usagers brutes globales** : 125 789
* **Lignes usagers écartées lors des filtres de cohérence initiaux** : 2 598 lignes.
* **Lignes après jointure large et nettoyage final (Silver)** : 125 829 (Le léger delta avec le fichier usager initial provient de l'alignement combinatoire de la jointure relationnelle avec l'entité véhicules).


* **Partitionnement de la silver** : La couche Silver a été écrite en Parquet sur le disque en appliquant un partitionnement par département : `.partitionBy("dep")`. Le département présente une cardinalité idéale (~100 modalités) et constitue le filtre de prédilection des analystes locaux. Cela active le mécanisme de **Partition Pruning** (élagage des partitions) lors des requêtes spatiales, évitant un scan complet du jeu de données.

---

## 3. Transformations et analyses métiers (silver -> gold)

### Analyse 1 - Agrégation (Météo vs Gravité)

* **Question** : Les conditions météorologiques ont-elles un impact direct sur la proportion d'accidents graves ?
* **Code clé** :

```python
analyse_meteo = (
    df_silver.withColumn("meteo", weather_desc)
    .groupBy("meteo")
    .agg(
        F.count("*").alias("total_impliques"),
        F.sum(F.when(F.col("grav") == 2, 1).otherwise(0)).alias("tues"),
        F.sum(F.when(F.col("grav") == 3, 1).otherwise(0)).alias("blesses_hospitalises"),
        F.round((F.sum(F.when(F.col("grav").isin(2, 3), 1).otherwise(0)) / F.count("*")) * 100, 2).alias("taux_gravite_pct")
    ).orderBy(F.desc("taux_gravite_pct"))
)

```

* **Résultat** :

```text
+-------------------+---------------+----+--------------------+----------------+
|meteo              |total_impliques|tues|blesses_hospitalises|taux_gravite_pct|
+-------------------+---------------+----+--------------------+----------------+
|Brouillard - fumée |509            |33  |131                 |32.22           |
|Autre/Non renseigné|482            |19  |110                 |26.76           |
|Temps éblouissant  |2180           |83  |483                 |25.96           |
|Vent fort - tempête|432            |18  |86                  |24.07           |
|Neige - grêle      |295            |8   |57                  |22.03           |
|Pluie forte        |3438           |100 |536                 |18.5            |
|Normale            |98497          |2633|15197               |18.1            |
|Temps couvert      |5017           |159 |704                 |17.2            |
|Pluie légère       |14979          |345 |1967                |15.43           |
+-------------------+---------------+----+--------------------+----------------+

```

* **Lecture métier** : Contre toute attente, le taux de gravité individuelle le plus élevé ne se rencontre pas sous la pluie forte ou la neige, mais par temps de **Brouillard - fumée (32,22%)** et lors d'un **Temps éblouissant (25,96%)**. Bien que la majorité absolue des accidents se produise sous une météo "Normale", la perte soudaine de visibilité ou l'aveuglement transitoire augmentent drastiquement la violence et la cinétique des chocs.

### Analyse 2 - Jointure (Catégories de Véhicules)

* **Question** : Quels sont les modes de transport les plus vulnérables en termes de létalité et d'hospitalisation ?
* **Code clé** :

```python
analyse_vehicule = (
    df_silver.withColumn("categorie_vehicule", vehicle_desc)
    .groupBy("categorie_vehicule")
    .agg(
        F.count("*").alias("total_impliques"),
        F.sum(F.when(F.col("grav") == 2, 1).otherwise(0)).alias("tues"),
        F.sum(F.when(F.col("grav") == 3, 1).otherwise(0)).alias("blesses_hospitalises"),
        F.round((F.sum(F.when(F.col("grav").isin(2, 3), 1).otherwise(0)) / F.count("*")) * 100, 2).alias("taux_gravite_pct")
    ).orderBy(F.desc("taux_gravite_pct"))
)

```

* **Résultat** :

```text
+------------------------+---------------+----+--------------------+----------------+
|categorie_vehicule      |total_impliques|tues|blesses_hospitalises|taux_gravite_pct|
+------------------------+---------------+----+--------------------+----------------+
|Deux-roues motorisé     |20413          |769 |6341                |34.83           |
|Bicyclette              |5463           |206 |1236                |26.4            |
|Autre / Inconnu         |6312           |171 |1117                |20.41           |
|Voiture (VL)            |79519          |1884|9508                |14.33           |
|Utilitaire / Poids Lourd|11888          |324 |961                 |10.81           |
|Transports en commun    |2234           |44  |108                 |6.8             |
+------------------------+---------------+----+--------------------+----------------+

```

* **Lecture métier** : Les usagers de **Deux-roues motorisés** affichent une vulnérabilité critique : **34,83%** d'entre eux finissent tués ou hospitalisés lors d'un impact (soit plus de 1 usager sur 3). Les cyclistes suivent à **26,4%**. À l'inverse, la structure déformable des Voitures Légères (VL) offre une protection supérieure, stabilisant leur taux à **14,33%**, tandis que les Transports en commun s'affirment comme le mode le plus sûr (**6,8%**).

### Analyse 3 - Window Function (Top 3 Départements Mensuels)

* **Question** : Quels sont les trois départements en tête de la sinistralité en France pour chaque mois de l'année ?
* **Code clé** :

```python
from pyspark.sql.window import Window

fenetre_mois = Window.partitionBy("mois").orderBy(F.desc("nb_accidents"))

top_departements = (
    accidents_par_dep_mois
    .withColumn("rang", F.row_number().over(fenetre_mois))
    .filter(F.col("rang") <= 3)
    .orderBy("mois", "rang")
)

```

* **Résultat** :

```text
+----+---+------------+----+
|mois|dep|nb_accidents|rang|
+----+---+------------+----+
|1   |75 |353         |1   |
|1   |92 |201         |2   |
|1   |93 |194         |3   |
|... |...|...         |... |
|6   |75 |519         |1   |
|6   |92 |279         |2   |
|6   |93 |264         |3   |
|... |...|...         |... |
|8   |75 |271         |1   |
|8   |13 |166         |2   |
|8   |93 |166         |3   |
+----+---+------------+----+

```

* **Lecture métier** : Le département de **Paris (75)** occupe invariablement la première place du classement, et ce, sur l'intégralité des 12 mois de l'année, suivi par ses voisins directs de la petite couronne (**92** et **93**). L'unique rupture structurelle intervient au mois d'**août**, où le département des **Bouches-du-Rhône (13)** se hisse à la 2e position nationale. Ce phénomène illustre l'impact des flux migratoires estivaux vers le littoral méditerranéen.

---

## 4. Optimisation mesurée

* **Optimisation choisie** : **Broadcast Join** (Jointure par diffusion).
* **Pourquoi** : Lors de la jointure entre notre table de faits Silver (125k lignes) et notre table de dimension référentielle des départements (107 lignes), l'utilisation d'une jointure standard force un échange de données (*Shuffle*) sur l'ensemble du réseau pour regrouper les clés identiques. Diffuser la table de dimension à chaque exécuteur supprime intégralement l'étape d'échange pour la table de faits.
* **Mesure avant/après** :
* Temps d'exécution avec **Sort-Merge Join** (sans broadcast forcé) : **7,560 secondes**
* Temps d'exécution avec **Broadcast Join** : **4,550 secondes**
* **Gain de performance** : **+39,8%** de réduction du temps de traitement.


* **Analyse du plan physique** :
En observant le plan physique via `.explain()`, on remarque que le `BroadcastHashJoin` reste présent dans les deux exécutions. C'est l'action directe de l'**AQE** (Adaptive Query Execution) de Spark 4 qui prend le relais à l'exécution. Constatant que la table de dimension ne contient que 107 lignes, Spark réécrit dynamiquement le plan final en `BroadcastHashJoin` même quand on force la coupure du seuil automatique. Forcer manuellement le broadcast via l'instruction `F.broadcast()` sécurise l'arbre d'exécution en éliminant explicitement toute velléité d'échange (`Exchange`) coûteuse sur le cluster.

---

## 5. Lecture de la Spark UI

* **Job observé** : Le calcul de l'Analyse 3 (Top départements par mois via `Window function`).
* **Où se produit le shuffle** : L'étape d'échange (`Exchange hashpartitioning`) se manifeste à deux endroits clairs : lors de la phase d'agrégation du `groupBy("mois", "dep")` puis au moment du fenêtrage `Window.partitionBy("mois")`, forçant les données associées à un même mois à migrer vers les mêmes partitions de calcul.
* **Nombre de stages et de tasks** : Le graphe de dépendance (DAG) segmente ce traitement en **3 stages**, séparés par les deux shuffles. Le nombre de tâches du dernier stage (200) correspond à la valeur par défaut de `spark.sql.shuffle.partitions`, confirmant que le calcul final s'appuie sur le partitionnement distribué natif de Spark.

---

## 6. Exploration au-delà du cours

* **Piste choisie** : **UDF (User-Defined Function) vs. Fonction native Spark**.
* **Question** : Quel est l'impact réel sur les performances du passage par un bloc de code Python impératif face à une transformation SQL déclarative native ?
* **Protocole** : Nous avons créé une fonction segmentant les heures d'accidents en quatre périodes ("Matin", "Après-midi", "Soir", "Nuit"). Nous l'avons soumise à deux exécutions distinctes sur notre jeu complet de 125 829 lignes préalablement mis en cache : d'abord via une décoration `@F.udf` Python standard, puis via une expression native conditionnelle `F.when().otherwise()`.
* **Mesures** :
* Temps d'exécution avec **UDF Python standard** : **0,589 seconde**
* Temps d'exécution avec **Fonction native Spark** : **0,552 seconde**


* **Conclusion** : La fonction native se révèle **6,3% plus rapide** que l'UDF. Bien que cet écart paraisse minime sur notre volume actuel de 125k lignes, il confirme la présence du goulot d'étranglement lié à la sérialisation inter-processus (*JVM Java ⇄ Interpréteur Python Worker*). Le plan d'exécution affiche une étape de type `BatchEvalPython` pour l'UDF, ce qui oblige Spark à faire sortir les données de la JVM pour les évaluer en Python avant de les réimporter. La recommandation architecturale est indiscutable : les UDF Python classiques doivent être proscrites au profit des fonctions natives ou des `pandas_udf` (Arrow).

---

## 7. Ce qu'on a appris et limites

* **Ce qui a marché** : La transition vers l'architecture de stockage Médaillon unifiée sous format Parquet a permis de manipuler des relations complexes de manière transparente. Les fonctionnalités de fenêtrage (Window) se sont révélées d'une puissance impressionnante pour générer des classements statistiques complexes en quelques lignes de code lisibles.
* **Ce qui a bloqué** : Les spécificités d'exécution sous l'environnement local Windows ont provoqué d'importants blocages de droits d'écriture IO en raison de l'absence originelle des binaires Hadoop (`winutils.exe`). L'analyse 3 a également généré une erreur `AttributeError: module 'pyspark.sql.functions' has no attribute 'Window'`, ce qui nous a permis de comprendre que l'API de fenêtrage réside de manière indépendante au sein de `pyspark.sql.window` et non des fonctions d'agrégation standard.
* **Ce qu'on ferait avec plus de temps** :
1. **Analyse Géospatiale Avancée** : Exploiter les coordonnées de latitude/longitude converties pour générer un partitionnement spatial et cartographier les grappes d'accidents (*K-means clustering* ou cartes de chaleur via *Folium/GeoPandas*).
2. **Modélisation Prédictive (MLlib)** : Implémenter un algorithme de classification (ex: *Random Forest*) pour tenter de prédire l'indice de gravité (`grav`) d'un accident à partir de facteurs explicatifs comportementaux et structurels (vitesse maximale autorisée, présence de ceintures/casques, météo).



```