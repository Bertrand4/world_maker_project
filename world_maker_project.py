#!/usr/bin/env python
# coding: utf-8

# # <div style="border: 20px ridge crimson; font-size:28px; text-decoration:underline; background-color:black; padding:25px 0 25px 0"><font color="white"><center>ÉTUDE GÉNÉRALISÉE DE LA SOUS-NUTRITION DANS LE MONDE</center></font></div>
# <div style="text-align:right; margin-right: 10px; margin-top:-15px"><I>Par Bertrand Delorme, Data Analyst</I></div><BR/>

# <div style="text-align:justify"><font color="crimson" size=3><B><U>INTRODUCTION:</U></B></font> <font size=3>Notre projet traitera de la faim dans le monde. Nous objecterons d'analyser, de cibler et de comprendre les différentes causes et besoins de ce fléau qui touche encore aujourd'hui une part importante de la population mondiale. Notre fil conducteur sera donc la problématique et le plan suivants:</font></div>
# <BR/>
# 
# <center><font color="crimson" size=4><B>Quelles sont les conditions, les causes et les enjeux de la faim dans le monde?</B></font></center>
# 
# <ul><li><font color="midnightblue">Mission 1 :</font> Nous procéderons dans une premier temps à la récolte des données et à une présentation graphique de chaque variable étudiée.</li> <li><font color="darkgreen">Mission 2 : </font>Nous objecterons ensuite de mettre les variables récoltées en relation et d'expliquer les causes de la faim dans le monde.</li> <li><font color="darkgoldenrod">Mission 3 : </font>Nous chercherons ensuite à regrouper les pays étudiés selon les variables présentées et d'en étudier les différents types de besoins.</li> <li><font color="darkred">Mission 4 :</font> Enfin, nous présenterons des axes d'amélioration en vue de lutter contre la faim dans le monde à différents niveaux.</li></ul>
# 
# <BR/>
# <div style="text-align:justify"><B>Les données que nous traiterons sont issues de la <a href="http://www.fao.org/faostat/fr/#data" target="_blank">Food and Agriculture Organization Corporate Statistical Database</a>, branche de l'ONU, tirées des bilans alimentaires présents sur leur site officiel. Au cours de ce projet, nous nous appuierons sur les chiffres de 2017 pour avoir une estimation de la situation actuelle mondiale. Nous nous concentrerons sur les états dont la FAO recense des personnes en sous-nutrition, en considérant ce taux comme nul pour les états dont la FAO n'en recense pas afin de mieux traîter nos données.</B></div>
# <BR/><BR/><BR/>

# <div style="border: 2px solid black; padding-top:0px; padding-bottom:20px">
#     
# <div style="border: 2px solid black; background-color:black; color:white; padding:20px ; margin:0px; font-size:35px"><center><B>Sommaire</B><a id="sommaire"></a></center></div>
# <div style="margin-top:20px; margin-bottom:40px"><font color="black" size=3px><center><I><B>Un code de couleurs à été attribué aux différentes parties à titre de repère.</B></I></center></font></div>
# 
#     
# <div style="display: flex; justify-content: flex-start; flex-wrap:wrap">
#     
# <div style="width:40%; margin-left:6%">
# <a href="#m1" style="text-decoration:none"><div style="border: 3px solid darkblue; padding: 10px ; margin:10px">
# <font color="darkblue" size=4px ><B><center>Mission 1 : Récolte des données et</center><center>mise au point sur la situation actuelle</center></B></font></div></a></div>
# 
# <div style="margin-left:5%; margin-top:10px">
# <a href="#m11"><font color="darkblue" size=3px><B>I. État actuel de la sous-nutrition dans le monde</B></font></a>
# <li><a href="#m111"><font color="mediumblue">I.1. Population et sous-nutrition</font></a></li>
# <li><a href="#m112"><font color="mediumblue">I.2. Urgence alimentaire : Deux types de problème</font></a></li>
# <li style="margin-bottom:5px"><a href="#m113"><font color="mediumblue">I.3. Prévisions de la sous-nutrition d'ici 2050</font></a></li>
# 
# <a href="#m12"><font color="darkblue" size=3px><B>II. Disposition et nécessité alimentaire</B></font></a>
# <li><a href="#m121"><font color="mediumblue">II.1. Besoins nutritionnels du corps humain et nécessité mondiale</font></a></li> 
# <li><a href="#m122"><font color="mediumblue">II.2. Disponibilité alimentaire mondiale</font></a></li>
# <li style="margin-bottom:5px"><a href="#m123"><font color="mediumblue">II.3. Disponibilité alimentaire destinée à l'homme</font></a></li>
# 
# <a href="#m13"><font color="darkblue" size=3px><B>III. Potentiel alimentaire</B></font></a>
# <li><a href="#m131"><font color="mediumblue">III.1. Définitions et standard</font></a></li>
# <li><a href="#m132"><font color="mediumblue">III.2. Potentiel seuil</font></a></li>
# <li style="margin-bottom:5px"><a href="#m133"><font color="mediumblue">III.3. Cartographie des potentiels alimentaires</font></a></li>
# 
# <a href="#m14"><font color="darkblue" size=3px><B>IV. Finalisation du DataFrame principal</B></font></a>
# <li><a href="#m141"><font color="mediumblue">IV.1. Disponibilité alimentaire synthétisée</font></a></li>
# <li><a href="#m142"><font color="mediumblue">IV.2. PIB par habitant</font></a></li>
# <li><a href="#m143"><font color="mediumblue">IV.3. Récapitulatif du DataFrame world</font></a></li></div></div>
# 
# 
# <div style="display: flex; justify-content: flex-start; flex-wrap:wrap; margin-top:25px">
#     
# <div style="width:40%; margin-left:6%">
# <a href="#m2" style="text-decoration:none"><div style="border: 3px solid darkgreen; padding: 10px ; margin:10px">
# <font color="darkgreen" size=4px><B><center>Mission 2 : Analyse des variables et</center><center>explications de la sous-nutrition</center></B></font></div></a></div>
# 
#     
# <div style="margin-top:5px; margin-left:5%">
# <a href="#m21"><font color="darkgreen" size=3px><B>I. Analyse des corrélations</B></font></a>
# <li><a href="#m211"><font color="forestgreen"> I.1. Matrice des corrélations</font></a></li>
# <li style="margin-bottom:5px"><a href="#m212"><font color="forestgreen"> I.2. Cercle des corrélations</font></a></li>
# 
# <a href="#m22"><font color="darkgreen" size=3px><B>II. Analyse de la variance de la sous-nutrition</B></font></a>
# <li><a href="#m221"><font color="forestgreen"> II.1. Préparatifs des analyses</font></a></li>
# <li><a href="#m222"><font color="forestgreen"> II.2. Variance de la sous-nutrition</font></a></li>
# <li><a href="#m223"><font color="forestgreen"> II.3. Variance du pourcentage de la sous-nutrition</font></a></li></div></div>
# 
# <div style="display: flex; justify-content: flex-start; flex-wrap:wrap; margin-top:25px">
# 
# <div style="width:40%; margin-left:6%">
# <div style="border: 3px solid darkgoldenrod; padding: 10px ; margin:10px">
# <a href="#m3" style="text-decoration: none"><font color="darkgoldenrod" size=4px><B><center>Mission 3 : Ciblage des zones dans</center><center>le besoin alimentaire</center></B></font></a></div></div>
#     
# <div style="margin-top:5px; margin-left:5%">
# <a href="#m31"><font color="darkgoldenrod" size=3px><B>I. Classification hiérarchique des états</B></font></a>
# <li><a href="#m311"><font color="goldenrod">I.1. Dendrogramme</font></a></li>
# <li><a href="#m312"><font color="goldenrod">I.2. Clustering et projection sur le 1er plan factoriel</font></a></li>
# <li style="margin-bottom:5px"><a href="#m313"><font color="goldenrod">I.3. Description des clusters</font></a></li>
# 
# <a href="#m32"><font color="darkgoldenrod" size=3px><B>II. Analyses graphiques interclusters</B></font></a>
# <li><a href="#m321"><font color="goldenrod">II.1. Fonctions préliminaires</font></a></li>   
# <li><a href="#m322"><font color="goldenrod">II.2. Représentations des clusters à l'échelle mondiale</font></a></li>
# <li><a href="#m323"><font color="goldenrod">II.3. Des potentiels alimentaires irréguliers</font></a></li>
# <li style="margin-bottom:5px"><a href="#m324"><font color="goldenrod">II.4. Inégalités économiques</font></a></li>
#     
# <a href="#m33"><font color="darkgoldenrod" size=3px><B>III. Observations approfondies</B></font></a>
# <li><a href="#m331"><font color="goldenrod">III.1. Besoins alimentaires</font></a></li>
# <li><a href="#m332"><font color="goldenrod">III.2. L'Inde et la Chine</font></a></li>
# <li><a href="#m333"><font color="goldenrod">III.3. Des pays dans l'urgence alimentaire</font></a></li></div></div>
# 
# 
# <div style="display: flex; justify-content: flex-start; flex-wrap:wrap; margin-top:25px">
#     
# <div style="width:40%; margin-left:6%">
# <a href="#m4" style="text-decoration:none"><div style="border: 3px solid darkred; padding: 10px ; margin:10px">
# <font color="darkred" size=4px><B><center>Mission 4 : Axes de lutte </center><center>contre la faim dans le monde</center></B></font></div></a></div>
#     
# 
# <div style="margin-left:5%; margin-top:5px">
# <a href="#m41"><font color="darkred" size=3px><B>I. Utilisation de la disponibilité alimentaire</B></font></a>
# <li><a href="#m411"><font color="crimson">I.1. Remaniement du DataFrame et nouvelles variables</font></a></li>
# <li><a href="#m412"><font color="crimson">I.2. Ajustement de constantes et fonction de la barre nutritive</font></a></li>
# <li style="margin-bottom:5px"><a href="#m413"><font color="crimson">I.3. Taux d'utilisation de la disponibilité alimentaire</font></a></li>
# 
# <a href="#m42"><font color="darkred" size=3px><B>II. Pertes alimentaires</B></font></a>
# <li><a href="#m421"><font color="crimson">II.1. Potentiel alimentaire pouvant être libéré dans le monde</font></a></li>
# <li><a href="#m422"><font color="crimson">II.2. Ciblage géographique</font></a></li>
# <li style="margin-bottom:5px"><a href="#m423"><font color="crimson">II.3. Des pays autosuffisants sans les pertes</font></a></li>
# 
# <a href="#m43"><font color="darkred" size=3px><B>III. Autres libérations alimentaires</B></font></a>
# <li><a href="#m431"><font color="crimson">III.1. Nourriture pour animaux</font></a></li>
# <li><a href="#m432"><font color="crimson">III.2. Autres utilisations</font></a></li>
# <li style="margin-bottom:5px"><a href="#m433"><font color="crimson">III.3. Taux d'exportations</font></a></li>
#     
# <a href="#m44"><font color="darkred" size=3px><B>IV. Conclusion</B></font></a></div></div>
# </div>

# <BR/>
# 
# ### <center>Nous importons dans un premier temps l'ensemble des librairies nécessaires pour l'ensemble du projet.</center>

# In[1]:


import pandas as pd ; import numpy as np ; import random ; import datetime as dt ; from functions import * ; import os
import matplotlib.pyplot as plt ; from matplotlib.pyplot import figure ; from matplotlib.pylab import arange, plot 
import matplotlib as mpl ; from pylab import * ; import matplotlib.patches as mpatches
from collections import Counter ; from termcolor import colored

import statsmodels.api as sm ; import statsmodels.formula.api as smf ; from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf ; from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose ; from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import* ; from statsmodels.tsa.api import ExponentialSmoothing

from sklearn import decomposition, preprocessing ; from sklearn.preprocessing import StandardScaler ; sc = StandardScaler()
from sklearn.cluster import KMeans ; from sklearn.metrics import make_scorer, r2_score ; from sklearn.svm import SVC
import scipy ; import scipy.stats as st ; from scipy.stats import t, shapiro, ks_2samp
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# La fonction dim() nous permettra de définir à chaque fois les dimensions des figures que l'on affichera:
def dim(x, y): return(plt.figure(figsize=(x, y)))

# Outils de génération des cartes du monde colorisées:
import cartopy ; import cartopy.io.shapereader as shpreader ; import cartopy.crs as ccrs
shpfilename = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
reader = shpreader.Reader(shpfilename) ; from world_cartography import *

from IPython.core.display import HTML ; import warnings ; warnings.filterwarnings('ignore')
HTML("""<style> .output_png {display: table-cell; text-align: center; vertical-align: middle} </style>""")


# Vous trouverez également ici <a href="https://bertrand4.github.io/world_maker_project/WM_packages" target="_blank">l'annexe</a> des modules importés, notamment la description des fonctions utilisées pour générer les cartes mondiales à venir.
# 
# 
# Vous noterez que j'ai tendance à "condenser" mes lignes de codes, ceci pour avoir une meilleure visibilité globale de mes cellules. Néanmoins, si cela vous pose problème pour la lecture, je vous invite à vous rediriger vers <a href="https://bertrand4.github.io/world_maker_project/world_maker_project_sp" target="_blank">ce script</a>.

# <BR/>
# <BR/>
# <BR/>
#     
# # <div style="border: 10px ridge darkblue; padding:15px; margin:5px"><a id="m1"><font color="darkblue"><center>Mission 1 : Récolte des données et mise au point sur la situation actuelles</center></font></a></div>

# # <a id="m11"><font color="darkblue"><U>I. État actuel de la sous-nutrition dans le monde</U></font></a>

# ## <a id="m111"><font color="mediumblue" style="margin-left:60px"><U><I>I.1. Population et sous-nutrition</I></U></font></a>

# <font color="midnightblue"><div style="text-align:justify"><B>Sur les 195 états du monde reconnus par l'ONU, seuls 166 seront pris en compte dans notre étude. Certaines des données que nous étudierons n'ont pas été relevées par la FAO pour les pays restants, c'est pourquoi nous nous limiterons notre étude aux 166 pays dont nous disposons de valeurs certifiées et en écarterons les autres pour un meilleur ajustement des résultats à venir.</B></div>

# <font color="midnightblue">Nous présentons donc notre DataFrame principal <font color="red"><B>world</B></font>, où seront collectées les données relevées sur la FAO, qui présente pour le moment les variables:
# - <font color="crimson"><I>population</I></font> : Le nombre d'habitants par pays.
# - <font color="crimson"><I>undernourished</I></font> : Le nombre d'habitants en sous-nutrition dans un pays donné.
# - <font color="crimson"><I>und_percentage</I></font> : Le pourcentage de la population en sous-nutrition au sein du pays donné.

# In[13]:


world = pd.read_csv("data/projet_R/W/world.csv").set_index("country")
und = world.loc[world["undernourished"] != 0] # Le DataFrame "und" ne contiendra que les pays recensant la sous-nutrition.

print("Aujourd'hui, %s états sur %d recensent une partie de la population la sous-nutrition." %(len(und), len(world)))
world


# <font color="midnightblue">On compte en 2017 environ __7,4 milliards d'habitants sur Terre__, dont plus de __850 millions de personnes en sous-nutrition__.

# In[6]:


POPULATION = sum(world["population"]) # Le nombre d'êtres humains sur Terre en 2017.
SSN = world["undernourished"].sum() # Nombre d'individus sur Terre en sous-nutrition.

dim(5, 5); mpl.rcParams["font.size"] = 14
plt.pie([POPULATION-SSN, SSN], labels = ["Suffisance nutritionnelle", "Sous-nutrition"], 
        colors = ["darkkhaki", "crimson"], explode = [0, 0.3], autopct = '%1.2f%%', shadow = True)

plt.title("Taux de la sous-nutrition mondiale", fontsize = 25)
plt.ylabel(""), plt.axis("equal"); plt.show()


# <font color="midnightblue">Ce graphique représente tout simplement le taux de la population mondiale en sous-nutrition, où l'on constate qu'environ <B>10,5%</B> du monde est encore aujourd'hui concernée par la famine. <font color="crimson"><B>La sous-nutrition concerne donc encore aujourd'hui plus d'un être humain sur 10.</B></font>
# <BR>
# 
# Un des objectifs de ce projet est de cibler les zones géographiques où l'on recense la famine, et d'en étudier les conditions et autres critères, c'est pourquoi nous allons générer une première carte du monde colorisée selon les différents pourcentages de la population locale d'un pays souffrant de la sous-nutrition.
# 
# <font color="midnightblue"><B>Par défaut, sur chaque carte que nous tracerons, les pays dont nous ne disposions pas des données nécessaires à leur études, et donc écartés du projet, resteront en blanc et ne seront pas à prendre en compte pour une quelconque interprétation. Ces cartes sont générés grâce au module cartopy importé en introduction, ainsi qu'à la fonction cartography() définie en <a href="https://bertrand4.github.io/world_maker_project/WM_packages" target="_blank">annexe</a> du projet.</B>

# In[35]:


def Sn(a, b): return(mapzone( world.loc[(world["und_percentage"] >= a) & (world["und_percentage"] < b)] ))
SN1, SN2, SN3, SN4, SN5, SN6, SN7 = Sn(0, 1), Sn(1, 10), Sn(10, 20), Sn(20, 30), Sn(30, 40), Sn(40, 50), Sn(50, 100)
palette = ["seagreen", "darkkhaki", "olive", "crimson", "darkred", "darkslategray", "black"]

dim(22, 7); ax = carte(plt.axes(projection=ccrs.PlateCarree()))
cartography([SN1, SN2, SN3, SN4, SN5, SN6, SN7], palette, ax)

plt.title("Cartographie de la sous-nutrition mondiale", fontsize=40, pad=10)
plt.legend(legendes(palette), ["Sous-nutrition inférieure à 1%"] + ["Entre 1% et 10%"] +
                              ["Entre {0}% et {1}%".format(i, i+10) for i in range(10, 41, 10)] +
                              ["Sous-nutrition supérieure à 50%"], 
           loc="lower left", fontsize=13)  ; plt.show()


# <font color="midnightblue">On retrouve bien le clivage géographique entre les pays développés et autosuffisants majoritairement au Nord, et les pays en voie de développement ou sous-développés au Sud où l'on relève la sous-nutrition. Les graves problèmes de famine en Afrique apparaissent clairement sur cette carte, mais observons cependant que l'on recense la sous-nutrition dans certains pays pourtant bien développés tels que le Mexique ou encore l'Estonie. Cet aspect sera notamment approfondi en mission 3 lorsque nous décrirons les différents critères des zones de sous-nutrition.

# ## <a id="m112"><font color="mediumblue" style="margin-left:60px"><I><U>I.2. Urgence alimentaire : Deux types de problème</U></I></font></a>

# <font color="midnightblue">Au court de ce projet, nous allons nous confronter à deux critères plus ou moins liés de pays en difficulté:
# - <font color="dimgrey"><B>Les pays recensants le plus d'êtres humains en sous-nutrition.</B></font>
# - <font color="crimson"><B>Les pays présentant les plus forts pourcentages de la population locale en sous-nutrition.</B></font>
# 
# Nous présentons ci-dessous les 15 pays les plus enclins à chacun de ces deux critères.

# In[8]:


top15_qua = world.sort_values("undernourished", ascending=False).head(15)
top15_per = world.sort_values("und_percentage", ascending=False).head(15)

dim(20, 14); plt.subplot(2, 2, 1)
top15_qua["undernourished"].plot(kind = "bar", color = "slategray")
plt.title("Nombre d'êtres humains en sous-nutrition par état", fontsize=20, color="#393838", pad=10)
plt.xticks(fontsize=14, rotation=75) , plt.xlabel("")
plt.yticks(fontsize=15) , plt.ylabel("Centaines de millions d'habitants", fontsize=15)

plt.subplot(2, 2, 2)
top15_per["und_percentage"].plot(kind = "bar", color = "crimson")
plt.title("Pourcentage de la population en sous-nutrition par état", fontsize=20, color="crimson", pad=10)
plt.xlabel(""), plt.ylabel("Pourcentage", fontsize=15), plt.ylim(0,100)
plt.xticks(fontsize=14, rotation=75), plt.yticks(fontsize=15) ; plt.show()


# <font color="midnightblue">L'Asie couvre à elle seule plus de 58% de la population mondiale. C'est sur ce continent que l'on relève le plus d'êtres humains sous-alimentés comme le montre le premier histogramme avec près de 200 millions et 125 millions d'individus en sous-nutrition respectivement en Inde et en Chine, ces deux pays étant les plus peuplés du monde avec respectivement 1,353 et 1,393 milliards d'habitants.
# 
# Sur le deuxième histogramme, nous observons les pays présentant les pires difficultés alimentaires, il est à noter que tous ces pays à l'exception d'Haïti se situent en Afrique. On en relève notamment deux, la République Centrafricaine et la Zimbabwe, qui ne parviennent pas même à nourrir la moitié de leur population respective. Nous traiterons certaines données du traitement alimentaire internes aux pays observés ici.

# ## <a id="m113"><font color="mediumblue" style="margin-left:60px"><I><U>I.3. Prévisions de la sous-nutrition d'ici 2050</U></I></font></a>

# <font color="midnightblue">Chaque année, en moyenne, environ 800 millions d'êtres humains souffrent de famine. En relevant sur le site de la FAO les données de la sous-nutrition dans le monde jusqu'à aujourd'hui, on peut établir un graphique représentant l'évolution du nombre de personnes sous-alimentées depuis 1950. Un modèle de Holt-Winters peut nous donner une idées de la prédiction de la sous-nutrition d'ici 2050 en se basant sur les chiffres actuels.

# In[11]:


undernourishment = pd.read_csv("data/projet_R/evo.csv").set_index("year")

und_pred = pd.Series(ExponentialSmoothing(np.asarray(undernourishment["undernourished"]), seasonal_periods=2,
                                          trend="add", seasonal="add").fit().forecast(6), index=range(2025, 2051, 5))

und_pred[2020]=undernourishment.loc[2020, "undernourished"]
und_pred=und_pred.sort_index()

dim(15, 5); plt.axhline(10**9, linestyle="--", color="crimson", linewidth=1)

plt.plot(undernourishment["undernourished"], label="Population en sous-nutrition", color="deepskyblue", linewidth=2.5)
plt.plot(und_pred, label="Prédiction de la sous-nutrition", color="lime", linewidth=2.5)

plt.title("Prédiction de la sous-nutrition d'ici 2050", fontsize=30, pad=10)
plt.ylabel("Milliards d'habitants", fontsize=20), plt.ylim(500000000, 1200000000), plt.yticks(fontsize=15)
plt.xticks(range(1950, 2051, 5)), plt.xticks(fontsize=12), plt.legend(fontsize=20, loc="lower right") ; plt.show()


# <font color="midnightblue">Cette prédiction montre que si l'on ne fait rien, nous sommes susceptibles de dépasser le milliard d'êtres humains sur terre en sous-nutrition d'ici 30 ans. L'objectif de ce projet étant de proposer des solutions à la non-réalisation de cette prédiction, nous commencerons par un rappel sur les besoins nutritionnels journaliers moyens d'un être humain et d'une mise au point sur la disponibilité alimentaire actuelle.

# <BR/>
#     
# # <a id="m12"><font color="darkblue"><U>II. Disposition et nécessité alimentaire</U></font></a>

# ## <a id="m121"><font color="mediumblue" style="margin-left:60px"><I><U>II.1. Besoins nutritionnels du corps humain et nécessité mondiale</U></I></font></a>

# <font color="midnightblue">D'après le site <a href="https://www.futura-sciences.com/" target="_blank">Futura Sciences</a>, un être humain a besoin, en moyenne d'un apport journalier de 2250 kcalories, environ, soit un besoin de 821250 kcalories par an. D'après le site <a href="https://www.e-sante.fr/" target="_blank">e-santé</a>, un être humain a besoin, en moyenne d'un apport journalier de 0,055 kg, environ, soit un besoin de 20,075 kg par an. Sachant cela, en tenant compte de la population mondiale, on peut estimer le nombre de kilocalories ainsi que le nombre de kg de protéines nécessaires dans la disponibilité alimentaire mondiale afin de nourrir le monde entier.

# In[14]:


KILO_CAL = 821250 # Quantité annuelle moyenne en kilocalorie nécessaire à un être humain.
KILO_PROT = 20.075 # Quantité annuelle moyenne de kilogrammes de protéines nécessaire à un être humain.

KCAL_REQUIREMENT = POPULATION * KILO_CAL # Le nombre de kilocalories nécessaires pour nourrir la Terre entière.
PROT_REQUIREMENT = POPULATION * KILO_PROT # Le nombre de kilogrammes de protéines nécessaire pour nourrir la Terre entière.

print("En 2017, on compte %s d'êtres humains, ce qui demande la disponibilité alimentaire annuelle mondiale de:" %POPULATION)
print("- %e kilocalories." %KCAL_REQUIREMENT) ; print("- %e kilogrammes de protéines." %PROT_REQUIREMENT)


# ## <a id="m122"><font color="mediumblue" style="margin-left:60px"><U><I>II.2. Disponibilité alimentaire mondiale</I></U></font></a>

# <font color="midnightblue"><B>À l'échelle mondiale, disposons-nous de ressources et de productions suffisante pour nourrir le monde entier?</B> C'est bien sûr la première question à se poser lorsque l'on veut tenter de libérer le monde de la faim. C'est pourquoi nous allons comparer dans cette sous-partie la disponibilité alimentaire mondiale et la disponibilité alimentaire nécessaire pour nourrir le monde entier. Commençons par ajouter à <font color="red"><B>world</B><font color="midnightblue"> les deux variables suivantes tirées de la FAO pour un pays donné:
# - <font color="crimson"><I>domestic_supply_kcal</I></font> : La disponibilité intérieure annuelle totale en terme de kilocalories.
# - <font color="crimson"><I>domestic_supply_kgproteine</I></font> : La disponibilité intérieure annuelle totale en terme de kilogrammes de protéines.

# In[15]:


supply = pd.read_csv("data/projet_R/W/supply.csv").set_index("country")
world = world.join(supply) ; world.head(5)


# <font color="midnightblue">Comparons alors sur un histogramme la disponibilité alimentaire mondiale réelle et la disponibilité alimentaire nécessaire pour nourrir le monde entier.

# In[16]:


dim(14, 6), plt.bar(0.7, KCAL_REQUIREMENT, color="dimgrey", label="Nécessaire")
plt.bar(1.3, world["domestic_supply_kcal"].sum(), width=0.7, color="darkkhaki", label="Disponible")
plt.ylabel("kilocalories ( x10¹⁶ )", fontsize=18)  ; plt.xticks(fontsize=20) ; plt.yticks(fontsize=13)

plt.gca().twinx()
plt.bar(3.7, PROT_REQUIREMENT, color="dimgrey", label="Nécessaire")
plt.bar(4.3, world["domestic_supply_kgproteine"].sum(), width=0.7, color="darkkhaki", label="Disponible")
plt.ylabel("kilogrammes de protéines ( x10¹¹ )", fontsize=20)

plt.title("Une disponibilité alimentaire mondiale amplement suffisante", fontsize=30, color="crimson", pad=17)
plt.xticks(range(0, 5), ["","Quantité alimentaire mondiale en calories","","","Quantité alimentaire mondiale en protéines"])
plt.yticks(fontsize=13); plt.legend(loc="upper center", fontsize=27) ; plt.grid(False); plt.show()


# <font color="midnightblue">Tout l'intérêt du projet est ici: <B>le monde est excédentaire que ce soit en termes de calories ou en terme de protéines</B>. Sachant cela, <B>il est indéniable qu'une solution pour libérer le monde de la faim existe</B>. Bien entendu, malheureusement, la famine est un fléau que l'on pourra difficilement inhiber totalement en dépit des conflits politiques actuels ou à venir, mais nous montrerons que certaines réorganisations pourraient tout de même réduire fortement le nombre d'êtres humains en sous-nutrition.

# ## <a id="m123"><font color="mediumblue" style="margin-left:60px"><I><U>II.3. Disponibilité alimentaire destinée à l'homme</U></I></font></a>

# <font color="midnightblue">L'ensemble de la disponibilité alimentaire n'est pas uniquement consacrée à la nourriture. Nous pouvons relever par exemple la producion alimentaire destinée aux animaux, les pertes, les semences... C'est pourquoi nous ajoutons maintenant à <font color="red"><B>world</B><font color="midnightblue"> les deux variables suivantes pour un pays donné:
# - <font color="crimson"><I>food_supply_kcal</I></font> : La disponibilité intérieure annuelle totale en nourriture en terme de kilocalories.
# - <font color="crimson"><I>food_supply_kgproteine</I></font> : La disponibilité intérieure annuelle totale en nourriture en terme de kilogrammes de protéines.
# 
# Attention à ne pas confondre ces nouvelles variables avec les deux précédentes!

# In[17]:


food = pd.read_csv("data/projet_R/W/food.csv").set_index("country")
world = world.join(food) ; world.head(5)


# <font color="midnightblue">Nous avons vu que la disponibilité alimentaire pouvait nourrir plus du double de la population mondiale. Si l'on se limite à la disponibilité alimentaire dédiée à la nourriture, nous observerons des résultats tout aussi révélateurs.

# In[28]:


a = 100 * world["food_supply_kcal"].sum() / KCAL_REQUIREMENT
b = 100 * world["food_supply_kgproteine"].sum() / PROT_REQUIREMENT
c = 100 * (POPULATION-SSN) / POPULATION

dim(10, 2), plt.barh([1, 2, 3], [a, b, c], color = ["dimgrey", "darkkhaki", "cornflowerblue"])

plt.title("La sous-nutrition dans un monde excédentaire", fontsize=25)
plt.yticks([1, 2, 3], ["Potentiel nutritif mondial en protéines", "Potentiel nutritif mondial en calories",
                       "Population mondiale en autosuffisance alimentaire"], fontsize=17)
plt.xticks(range(0, 151, 20), ["{}%".format(i) for i in range(0, 151, 20)], fontsize=20) ; plt.show()


# <font color="midnightblue">Au bas mot, en ne tenant compte que de la disponibilité alimentaire destinée à la nourriture, nous disposons de quoi nourrir plus de 125% de la population mondiale, alors que près de 10,5% souffre de sous-nutrition. Il est évident que les relations internationales politiques et commerciales sont à l'origine de cette mauvaise répartition alimentaire, mais il est important d'affirmer que nous avons néanmoins, largement les capacités de nourrir le monde entier. Les pays en sous-nutrition ne sont pas tous nécessairement dans le "même besoin". Par exemple, certains nécessites un apport alimentaire plus riche en protéine malgré une autosuffisance en calories, d'autres inversement, etc... Mais nous pouvons relever que ces deux nouvelles variables sont sensiblement proportionnelles comme le montre le nuage suivant.

# In[29]:


# Nous retirons ces quelques outliers pour une meilleures visibilité :
df = world.drop(index = ["Chine", "Inde", "États-Unis", "Brésil", "Indonésie", "Russie"]) 

dim(15, 4), plt.plot(df["food_supply_kcal"], df["food_supply_kgproteine"], "o", color="cornflowerblue")
plt.title("Disponibilité en calories en fonction de la disponibilité en protéines", fontsize=25, pad=10)
plt.xlabel("Kilo calories (x10¹⁴)", color="saddlebrown", fontsize=20)
plt.ylabel("Kilogrammes de protéines (x10⁹)", color="#393838", fontsize=15) ;  plt.show()


# <font color="midnightblue">Nous pouvons donc estimer une certaine corrélation entre ces deux types de disponibilité alimentaire. C'est pourquoi nous allons créer dans la partie suivante une nouvelle variable qui donnera un indice des besoins de ces deux dernières.

# <BR/>
#     
# # <a id="m13"><font color="darkblue"><U>III. Potentiel alimentaire</U></font></a>

# ## <a id="m131"><font color="mediumblue" style="margin-left:60px"><U><I>III.1. Définitions et standard</I></U></font></a>

# <font color="midnightblue">Nous avons relevé jusqu'ici le nombre d'habitants dans chaque pays, la disponibilité intérieure alimentaire en terme de calories et de protéines ainsi que les besoins nutritionnels annuels du corps humain concernant ces deux unités d'énergie. Nous pouvons alors introduire deux nouvelles variables synthétiques dans <font color="red"><B>world</B> <font color="midnightblue">:
# - <font color="crimson"><I>potentiel_calorie</I></font> : Qui nous renvoie le pourcentage de la population que le pays en question est en mesure de nourrir en terme de calories.
# - <font color="crimson"><I>potentiel_proteine</I></font> : Qui nous renvoie le pourcentage de la population que le pays en question est en mesure de nourrir en termine de protéines.

# In[32]:


world["potentiel_calorie"] = 100 * world["food_supply_kcal"] / (world["population"] * KILO_CAL)
world["potentiel_proteine"] = 100 * world["food_supply_kgproteine"] / (world["population"] * KILO_PROT)
aut , und = world.loc[world["undernourished"] == 0] , world.loc[world["undernourished"] != 0]

fig, ax = plt.subplots(figsize=(7, 7))
plt.axhline(100, color="crimson") ; plt.axvline(100, color="crimson")

aut.plot.scatter(ax=ax, x="potentiel_calorie", y="potentiel_proteine", s=80, color="goldenrod", 
                 fig=fig, label="Pays autosuffisants")

und.plot.scatter(ax=ax, x="potentiel_calorie", y="potentiel_proteine", s=80, color="dimgrey",
                 fig=fig, label="Pays recensant la sous-nutrition")

plt.title("Potentiels alimentaires en calories et en protéines", fontsize=25, pad=15) 
plt.xticks(fontsize=15) ; plt.yticks(fontsize=15) ; plt.xlim(70, 180) ; plt.ylim(70, 250)
plt.xlabel("Potentiel nutritif en calories (%)", fontsize=20); plt.ylabel("Potentiel nutritif en protéines (%)", fontsize=20)
plt.legend(fontsize=15, loc="upper left") ; plt.show()


# <font color="midnightblue">Nous avons représenté sur ce graphique les états selon leur potentiel nutritif:
# - Les pays situés dans la partie supérieure droite sont les pays dont le potentiel alimentaire intérieur serait suffisant pour nourrir la population locale.
# - Dans la partie supérieure gauche, on y trouve les pays capables de nourrir leur population en terme de protéines mais pas en terme de calories.
# - Inversement dans la partie inférieure droite.
# - La partie inférieure gauche représente les pays dont les besoins sont des plus urgents, leur disponibilité intérieure ne leur permet pas de nourrir leur population, ni en protéines, ni en calories.
# 
# <font color="midnightblue">Nous pouvons nous demander s'il existe une corrélation entre chacun de ces deux types de manque et la position géographique des pays concernés, ce que nous allons visualiser sur la carte suivante, en considérant un potentiel alimentaire comme étant "insuffisant" sous le seuil de 110%. 

# In[37]:


WPC , WPP = world["potentiel_calorie"] , world["potentiel_proteine"]
P1 , P2 = mapzone(world.loc[(WPC > 110) & (WPP > 110)]) , mapzone(world.loc[(WPC > 110) & (WPP <= 110)])
P3 , P4 = mapzone(world.loc[(WPC <= 110) & (WPP > 110)]) , mapzone(world.loc[(WPC <= 110) & (WPP <= 110)])
palette = ["mediumaquamarine", "darkkhaki", "dimgray", "black"]

dim(22, 7) ; ax = carte(plt.axes(projection=ccrs.PlateCarree()))
cartography([P1, P2, P3, P4], palette, ax)

plt.title("Cartographie des potentiels alimentaires", fontsize=35, pad=15)
plt.legend(legendes(palette), ["Pays Autosuffisants", "Potentiel calories insuffisant", "Potentiel protéines insuffisant", 
                               "Potentiels calories et protéines insuffisant"], loc="lower left", fontsize=13) ; plt.show()


# <font color="midnightblue">On peut constater que les manques alimentaires se concentrent en Afrique Centrale et en Afrique du Sud. Mais il est intéressant de noter que les deux types de manque sont irréguliers selon les cibles géographiques. Considérons aussi que les pays d'Asie sont globalement autosuffisants alors que nous y avons relevé de lourdes populations sous-alimentées.

# ## <a id="m132"><font color="mediumblue" style="margin-left:60px"><U><I>III.2. Potentiel seuil</I></U></font></a>

# <font color="midnightblue">Dans le cadre des analyses à venir, nous chercherons à tenir compte de chaque pays ayant un potentiel alimentaire insuffisant dans l'une des deux unités d'énergie étudiée. C'est pourquoi nous ajouterons à <font color="red">__world__ <font color="midnightblue"> la variable synthétique suivante:
# - <font color="crimson"><I>potentiel</I></font> : Qui nous renvoie le minimum des deux variables précédentes, donc le pourcentage de la population que le pays en question est en mesure de nourrir en calories et en protéines.
#     
# C'est ce que nous appellerons <B>potentiel alimentaire</B> tout le long du projet.

# In[38]:


world["potentiel"] = world[["potentiel_calorie", "potentiel_proteine"]].min(axis=1)
world[["population", "und_percentage", "potentiel_calorie", "potentiel_proteine", "potentiel"]].head(5)


# <font color="midnightblue">Il serait donc intéressant de visualiser l'ensemble des pays étudiés selon leur potentiel alimentaire, à titre de comparaison.

# In[41]:


fig, axes = plt.subplots(figsize=(20, 6))
axes.axhline(100, linestyle="--", color="black", linewidth=0.8)

df1 = world.sort_values("potentiel", ascending=False).loc[world["potentiel"] >= 150]
df2 = world.sort_values("potentiel", ascending=False).loc[(world["potentiel"] < 150) & (world["potentiel"] >= 125)]
df3 = world.sort_values("potentiel", ascending=False).loc[(world["potentiel"] < 125) & (world["potentiel"] >= 100)]
df4 = world.sort_values("potentiel", ascending=False).loc[world["potentiel"] < 100]

leg = ["Pays excédentaires", "Pays autosuffisants", "Potentiel alimentaire fragile", "Potentiel alimentaire insuffisant"]
palette=["gold", "orange", "red", "darkred"]

for k, data in enumerate([df1, df2, df3, df4]) : axes.bar(data.index, data["potentiel"], color=palette[k], label=leg[k])
    
axes.set_title("Pourcentage de la population locale pouvant être nourrie", fontsize=35, pad=15)
plt.xlabel("Pays", fontsize=17) ; plt.xticks(fontsize=5, rotation=90)
axes.set_yticklabels(["{}%".format(int(x)) for x in axes.get_yticks()], fontsize=20)
axes.grid(True), axes.legend(fontsize=15) ; plt.show()


# <font color="midnightblue">Sur un deuxième histogramme, nous pouvons effectuer un zoom du premier sur les états au potentiel alimentaire insuffisant ainsi que sur les états dont le potentiel alimentaire dépasse 155%.

# In[17]:


fig, axes = plt.subplots(figsize=(15, 4))
axes.axhline(100, linestyle="--", color="black", linewidth=0.8) # États excédentaires et en sous-nutrition
axes.bar(df1.index, df1["potentiel"], color="gold", label="Pays excédentaires")
axes.bar(df4.index, df4["potentiel"], color="darkred", label="Pays au potentiel alimentaire insuffisant")
axes.set_title("Zoom sur les pays excédentaires et en sous-nutrition", fontsize=25) 
plt.xticks(fontsize=10, rotation=90) ; axes.set_yticklabels(["{}%".format(int(x)) for x in axes.get_yticks()], fontsize=15)
plt.grid(True), plt.legend(fontsize=15); plt.show()


# <font color="midnightblue"><div style="text-align:justify">Le complexe Nord/Sud apparaît encore une fois clairement sur ce diagramme: alors que plusieurs états d'Europe et d'Amérique du Nord sont en mesure de nourrir plus de 150% de leur population, on distingue notamment des d'Afrique Subsaharienne et d'Asie centrale ne pouvant subvenir aux besoins alimentaires de chacun de leurs habitants. <B>Cela fait clairement apparaître la cause de la faim dans le monde qui est bel et bien une distribution alimentaire mal répartie.</B></div>

# ## <a id="m133"><font color="mediumblue" style="margin-left:60px"><U><I>III.3. Cartographie des potentiels alimentaires</I></U></font></a>

# <font color="midnightblue">Enfin, visualisons ce déséquilibre alimentaire mondial sur une nouvelle carte, le complexe Nord/Sud y sera alors encore une fois bien visible.

# In[42]:


for i in range(1, 5) : vars()["Z"+str(i)]=mapzone(vars()["df"+str(i)])
    
dim(22, 7) ; ax = carte(plt.axes(projection=ccrs.PlateCarree()))
cartography([Z1, Z2, Z3, Z4], palette, ax)

plt.title("Un déséquilibre alimentaire mondial", fontsize=35, pad=10)
plt.legend(legendes(palette), ["Pays Excédentaires", "Pays Autosuffisants", "Potentiel Alimentaire Fragile",
                               "Potentiel Alimentaire Insuffisant"], loc="lower left", fontsize=14) ; plt.show()


# <font color="midnightblue"><div style="text-align:justify">Naturellement, on y retrouve les pays excédentaires parmi les états d'Europe et d'Amérique du Nord. Mais la corrélation évidente entre le potentiel alimentaire et le pourcentage de la population locale en sous-nutrition est biaisée par des pays comme le Japon et l'Australie, où l'ONU ne recense pas la famine mais où l'on observe un potentiel alimentaire fragile.</div>

# <BR/>
# 
# # <a id="m14"><font color="darkblue"><U>IV. Finalisation du DataFrame principal</U></font></a>

# ## <a id="m141"><font color="mediumblue" style="margin-left:60px"><I><U>IV.1. Disponibilité alimentaire synthétisée</U></I></font></a>

# <font color="midnightblue">Nous souhaitons cibler les différents besoins en calories et en protéines des pays en difficulté. Cependant, les deux types de disponibilités alimentaires semblent relativement corrélées comme le confirme le graphique suivant.

# In[43]:


dim(15,3), plt.plot(world["domestic_supply_kcal"], world["domestic_supply_kgproteine"], "o", color="slategrey")
plt.title("Disponibilité alimentaire", fontsize=20) ; plt.xlim(0, 2*10**14) ; plt.ylim(0, 10**10)
plt.xlabel("Kilocalories (x10¹⁴)", fontsize=15) ; plt.ylabel("Kilogrammes de protéines (x10¹⁰)", fontsize=13) ; plt.show()


# <font color="midnightblue">C'est pourquoi, pour l'étude à venir, nous ne considérerons qu'une seule variable représentant ces deux dernières, de façon relativement "impartiale" (ne favorisant aucune des deux unités d'énergie en particulier). Nous ajoutons donc à <font color="red"><B>world</B></font> la variable suivante:
# - <font color="crimson"><I>domestic_supply</I></font> : Qui nous renvoie tout simplement la disponibilité alimentaire en milliers de tonnes de nourriture.

# In[44]:


dss = pd.read_csv("data/projet_R/W/alim.csv").drop(columns="exportation").set_index("country")
world = world.join(dss) ; world[["population", "undernourished", "domestic_supply"]].head(5)


# <font color="midnightblue">Une analyse des corrélations justifiera cette simplification.

# In[45]:


df = world[["domestic_supply_kcal", "domestic_supply_kgproteine", "domestic_supply"]]
dim(5, 5), sns.heatmap(df.corr(), annot=True, vmax=.8, square=True, cbar="plasma") ; plt.axis("equal")
plt.title("Corrélations des variables de la disponibilité alimentaire", fontsize=20) ; plt.show()


# <font color="midnightblue">Les variables <I>domestic_supply_kcal</I> et <I>domestic_supply_kgproteine</I> seront donc tout simplement remplacées par <I>domestic_supply</I> dans les analyses à venir.

# ## <a id="m142"><font color="mediumblue" style="margin-left:60px"><I><U>IV.2. PIB par habitant</U></I></font></a>

# <font color="midnightblue">La sous-nutrition dans le monde relève bien sûr d'un problème d'ordre financier, mais nous verrons que ce n'est pas le seul facteur relatif à ce fléau. Bien que ce projet ait pour ambition de contourner au mieux cet aspect, nous pouvons néanmoins manifester les inégalités économiques dans le monde. C'est pourquoi nous ajoutons au DataFrame <font color="red">__world__ <font color="midnightblue">la nouvelle variable:
# - <font color="crimson"><I>pib_hbt</I></font> : Qui représente pour chaque pays le PIB par habitant basé sur le dollar américain.

# In[46]:


pib = pd.read_csv("data/projet_R/W/gdp.csv").set_index("country") ; world = world.join(pib)
world[["population", "undernourished", "pib_hbt"]].head(5)


# <font color="midnightblue">Nous allons maintenant étudier la répartition des moyens financiers par personne dans le monde au niveau de chaque pays, avec un nuage de points représentant le PIB par habitant en fonction de la population, puis avec une courbe de Lorenz sur la répartition du PIB par habitant dans chaque pays.

# In[48]:


dim(14, 10), plt.subplot(2, 2, 1); sci=world.drop(index=["Chine", "Inde"])
plt.plot(sci["population"], sci["pib_hbt"], "o", color="mediumseagreen")
plt.xlabel("Population (x10⁸)") ; plt.ylabel("PIB/habitant (dollar US)")
plt.title("Population d'un pays donné selon le PIB par habitant", fontsize=15)

plt.subplot(2, 2, 2); taux = world.sort_values("pib_hbt")["pib_hbt"].values; n = len(taux)
lorenz = np.append([0], np.cumsum(taux)/sum(taux)); gini=2*(0.5-lorenz.sum()/n)
X = arange(0,1,0.01); plt.plot(np.linspace(0-1/n, 1+1/n, n+1), lorenz) ; plot(X, X)
plt.title("Courbe de Lorenz - PIB/habitants mondial - Gini=%s" %(round(gini, 2)), fontsize=15) ; plt.show()


# <font color="midnightblue">Ce n'est une découverte pour personne que de dire que l'économie est mal répartie au niveau mondiale. La courbe de Lorenz présente un indice de Gini de 0.62 ce qui marque le fossé entre les pays excédentaires et les pays en sous-nutrition. Sur le graphique de gauche, on comprend que le PIB par habitant peut s'avérer faible pour des pays extrêmement peuplés. Si l'économie était répartie correctement, nous observerions une tendance linéaire croissante sur ce graphique, ce qui n'est pas le cas.

# ## <a id="m143"><font color="mediumblue" style="margin-left:60px"><I><U>IV.3. Récapitulatif du DataFrame WORLD</U></I></font></a>

# <font color="midnightblue"><B>En fin de compte de cette première partie introductive, notre DataFrame principal</B> <font color="red"><B>world</B><font color="midnightblue"><B> comprend les variables suivantes:</B>
# - <B>1°)</B> <font color="crimson"><B><I>population</I></B> <font color="midnightblue"><B>:</B> Le nombre d'habitants par pays.
# - <B>2°)</B> <font color="crimson"><B><I>undernourished</I></B> <font color="midnightblue"><B>:</B> Le nombre d'habitants en sous-nutrition au sein du pays donné.
# - <B>3°)</B> <font color="crimson"><B><I>und_percentage</I></B> <font color="midnightblue"><B>:</B> Le pourcentage de la population en sous-nutrition au sein du pays donné.
# - <B>4°)</B> <font color="crimson"><B><I>domestic_supply_kcal</I></B> <font color="midnightblue"><B>:</B> La disponibilité intérieure annuelle totale en terme de kilocalories.
# - <B>5°)</B> <font color="crimson"><B><I>domestic_supply_kgproteine</I></B> <font color="midnightblue"><B>:</B> La disponibilité intérieure annuelle totale en terme de kilogrammes de protéines.
# - <B>6°)</B> <font color="crimson"><B><I>food_supply_kcal</I></B> <font color="midnightblue"><B>:</B> La disponibilité intérieure annuelle totale en nourriture en terme de kilocalories.
# - <B>7°)</B> <font color="crimson"><B><I>food_supply_kgproteine</I></B> <font color="midnightblue"><B>:</B> La disponibilité intérieure annuelle totale en nourriture en terme de kilogrammes de protéines.
# -  <B>8°)</B> <font color="crimson"><B><I>potentiel_calorie</I></B> <font color="midnightblue"><B>:</B> Qui nous renvoie le pourcentage de la population que le pays en question est en mesure de nourrir en terme de calories.
# -  <B>9°)</B> <font color="crimson"><B><I>potentiel_proteine</I></B> <font color="midnightblue"><B>:</B> Qui nous renvoie le pourcentage de la population que le pays en question est en mesure de nourrir en terme de protéines.
# -  <B>10°)</B> <font color="crimson"><B><I>potentiel</I></B> <font color="midnightblue"><B>:</B> Qui nous renvoie le pourcentage de la population que le pays en question est en mesure de nourrir.
# - <B>11°)</B> <font color="crimson"><B><I>domestic_supply</I></B> <font color="midnightblue"><B>:</B> La disponibilité alimentaire totale en milliers de tonnes de nourriture.
# - <B>12°)</B> <font color="crimson"><B><I>pib_hbt</I></B> <font color="midnightblue"><B>:</B> Qui représente pour chaque pays le PIB par habitants basé sur le dollar américain.

# In[49]:


world


# <div style="text-align:right"><a href="#sommaire"><I>Retour au sommaire</I></a></div>

# <BR>
# <BR>
# <BR>
# 
# # <div style="border: 10px ridge darkgreen; padding:15px; margin:5px"><a id="m2"><font color="darkgreen"><center>Mission 2 : Analyse des variables et explication de la sous-nutrition</center></font></a></div>

# <div style="border: 2px ridge darkgreen; padding:5px; text-align:justify"><B><I>Cette deuxième mission présente un aspect très technique de l'analyse de données, moins accessible au lecteur lambda que le reste du projet. Si vous n'êtes pas familier avec les méthodes d'analyse en Python, passez directement à la mission 3.</I></B></div>

# 
# 
# # <a id="m21"><font color="darkgreen"><U>I. Analyse des corrélations</U></font></a>

# ## <a id="m211"><font color="forestgreen" style="margin-left:60px"><I><U>I.1. Matrice des corrélations</U></I></font></a>

# <font color="darkgreen">L'objectif de cette deuxième mission est d'expliquer la sous-nutrition selon les variables étudiées. Nous rappelons que l'on distingue deux types de problème dans ce sujet:
# - Les pays recensants le plus d'êtres humains en sous-nutrition donnés par la variable <font color="red"><B><I>undernourished</I></B></font>.
# - Les pays présentant les plus forts pourcentages de la population locale en sous-nutrition donnés par la variable <font color="red"><B><I>und_percentage</I></B></font>.
# 
# <font color="darkgreen">Analysons les corrélations entre chacune des variables étudiées pour le clustering, deux à deux, avec un tableau de contingence coloré, et observons les corrélations liées à la population ainsi qu'aux deux variables de la sous-nutrition.

# In[50]:


df = world.drop(columns=["domestic_supply_kcal", "domestic_supply_kgproteine", "potentiel_calorie", "potentiel_proteine"])
dim(10, 8), sns.heatmap(df.corr(), annot=True, vmax=.8, square=True, cbar="plasma")
plt.title("Matrice des corrélations - WORLD", fontsize=20) ; plt.axis("equal") ; plt.show()


# <center><B>Observations</B></center>
# <BR>
# 
# <font color="black"><B>Population:</B> <font color="darkgreen">On remarque tout d'abord <font color="sienna">une très faible corrélation entre le nombre d'habitants d'un pays donné (population) et son pourcentage de sous-nutrition. <font color="darkgreen">La Chine et l'Inde sont deux exceptions de par leur population excessivement grande par rapport au reste du monde, mais l'on en déduit d'un point de vue général que ce ne sont pas forcément les pays les plus peuplés qui souffrent le plus de la sous-nutrition à l'échelle locale. En revanche, <font color="darkblue">la population est naturellement fortement corrélée à chaque variable donnant une estimation de la disponibilité alimentaire du pays<font color="darkgreen">. Plus un pays est peuplé, plus il est productif au niveau de l'agriculture et de la production alimentaire.<BR><BR>
#     
# <font color="darkgreen"><font color="black"><B>Sous-nutrition:</B></font> 
# - Tout d'abord, il est intéressant d'observer <font color="sienna">la faible corrélation entre les variables "undernourished" et "und_percentage"</font>, ce qui signifie que ce n'est pas spécialement dans les pays où le pourcentage de sous-nutrition est particulièrement élevé que l'on y trouve un nombre important d'êtres humains souffrants de la famine (ce qui, quelque part, est plus ou moins une bonne nouvelle).<BR><BR>
# - La variable "undernourished" est naturellement <font color="darkred">corrélée fortement négativement au PIB par habitants.<font color="darkgreen"> et <font color="darkred">très fortement négativement au potentiel du pays à nourrir son peuple<font color="darkgreen">. Sans surprise, ce sont les pays pauvres et/ou au potentiel nutritif insuffisant qui souffrent le plus de la famine.<BR><BR>
# - Il est cependant intéressant de relever que <font color="darkred">la corrélation avec le taux d'importation est négative<font color="darkgreen">. Les exportations sont moins courantes vers les pays souffrant de sous-nutrition, alors que la production n'y est pas particulièrement absente. Nous verrons que, pour certains pays, la nourriture est utilisée à des fins économiques plutôt qu'à des fins nutritives en dépit d'une population affamée.

# ## <a id="m212"><font color="forestgreen" style="margin-left:60px"><I><U>I.2. Cercle des corrélations</U></I></font></a>

# <font color="darkgreen">Réalisons maintenant une ACP sur nos données afin d'analyser l'inertie des variables regroupées sur un cercle des corrélations. Nous pourrons alors dégager certaines estimations sensiblement liées entre deux variables pour un pays donné. Nous ne nous intéresserons qu'au premier plan factoriel.

# In[52]:


X = df.values ; n_comp = 8
std_scaler = preprocessing.StandardScaler().fit(X)
X_scaled = std_scaler.transform(X)

data_pca = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
data_pca = data_pca.fillna(data_pca.mean())
names , features = data_pca.index , data_pca.columns
pca = decomposition.PCA(n_components=n_comp) ; pca.fit(X_scaled) ; pcs = pca.components_

# Présentons les deux nouvelles variables créées dans le tableau appelé "composantes":
composantes=pd.DataFrame(pca.fit_transform(X_scaled), index=world.index)
composantes.columns=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

display_circles(pcs, n_comp, pca, [(0,1)], labels = np.array(features))


# <font color="darkgreen">On retrouve bien la non-corrélation entre les variables <B>undernourished</B> et <B color="yellow">und_percentage</B>, dont les vecteurs sont quasiment orthogonnaux. Il est intéressant de voir que l'ensemble des variables autour du nombre d'êtres humains en sous-nutrition forment plus de 50% de l'inertie totale, tandis que près de 30% s'explique par des variables corrélées au pourcentage de sous-nutrition. C'est en se basant sur le cercle des corrélations ci-dessus que nous choisirons les variables explicatives pour une ANOVA sur chacune de ces deux variables, qui nous permettra maintenant de les expliquer plus spécifiquement.

# <BR/>
# 
# # <a id="m22"><font color="darkgreen"><U>II. Analyse de la variance de la sous-nutrition</U></font></a>

# ## <a id="m221"><font color="forestgreen" style="margin-left:60px"><I><U>II.1. Préparatifs des analyses</U></I></font></a>

# <font color="darkgreen">On s'intéresse aux variables <font color="crimson">___undernourished___<font color="darkgreen"> et <font color="crimson">___und_percentage___<font color="darkgreen"> du DataFrame <font color="red">__world__<font color="darkgreen"> et nous présentons en premier lieu les mesures de tendance centrale de ces variables dans une boîte à moustaches.

# In[53]:


# On définit ici les paramètres des boxplots pour toute la suite du projet:
medianprops , meanprops = {"color":"black"} , {"marker":"o", "markeredgecolor":"black", "markerfacecolor":"firebrick"}

dim(18, 2), plt.boxplot(world["undernourished"], showfliers=False, medianprops=medianprops, 
                        vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.title("Nombre d'êtres humains en sous-nutrition par état", fontsize=25) ; plt.show()

dim(18, 2), plt.boxplot(world["und_percentage"], showfliers=False, medianprops=medianprops, 
                        vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.title("Pourcentage de la sous-nutrition par état", fontsize=25) ; plt.show()


# <font color="darkgreen">Nous réaliserons chaque test statistique au niveau de 5%. On rappelle dans la cellule suivante notre DataFrame <font color="red">__world__<font color="darkgreen"> décrit à la fin de la mission précédente, et on en profite pour fixer notre niveau de test nt pour toute modélisation qui s'exécutera dans toute la suite du projet. 
#     
# Pour l'étude du nombre de personnes en sous-nutrition: Si nous prenions en compte l'Inde et la Chine, cela pourrait fausser nos données. En effet, nous pouvons observer sur l'ensemble des graphiques jusqu'à maintenant que ces deux pays se distinguaient de par leur population. C'est pourquoi nous considérerons à part ces deux états, afin d'estimer d'autres variables que la taille de la population pour expliquer la sous-nutrition, en les excluant de l'analyse de la variance.
# <BR/><BR/>
# Pour l'étude des pourcentages de la sous-nutrition, nous réduirons les données en pourcentage afin de comparer des grandeurs similaires.

# In[54]:


nt = 0.05 # Fixation du niveau de test statistique pour l'ensemble du projet.

world_num = world.drop(index=["Chine", "Inde"])
world_num["pib"]=world_num["population"]*world_num["pib_hbt"]
world_num["potentiel"]*=world_num["population"]/100

world_per = world.copy()
world_per["population"]/=(0.01*POPULATION)
DS=world["domestic_supply"].sum()
world_per["domestic_supply"]/=(0.01*DS)


# <font color="darkgreen">En vue des analyses répétées que nous exécuterons sur les __résidus des modélisations__ <font color="darkgreen">à venir, nous créons ici les fonctions décrites ci-dessous qui prendront toutes en entrée une modélisation donnée pour renvoyer un certain résultat sur ses résidus. À noter que les différents tests d'adéquation à une loi gausienne que nous exécuterons testeront donc l'adéquation d'une certaine variable X à la loi normale de paramètres (mean(X), std(X)).
#     
# - 1°) <font color="navy">__"residus_henry"__<font color="darkgreen">: qui nous superposera les résidus d'une modélisation donnée avec la droite de Henry.
# - 2°) <font color="navy">__"nuage_residuel"__<font color="darkgreen">: qui nous renverra le nuage de la variance résiduelle.
# - 3°) <font color="navy">__"analyse des résidus"__<font color="darkgreen">: qui nous renverras la droite de Henry ET le nuage de la variance résiduelle.
# - 4°) <font color="navy">__"test_shapiro"__<font color="darkgreen">: qui nous renverra les résultats d'un test de Shapiro sur les résidus (peu probable qu'on s'en serve vu la taille des échantillons).
# - 5°) <font color="navy">__"test_residus_KS"__<font color="darkgreen">: qui nous renverra les résultats d'un test de Kolmogorov-Smirnoff sur les résidus.
# - 6°) <font color="navy">__"test_residus_BP"__<font color="darkgreen">: qui nous renverra les résultats d'un test de Breusch-Pagan sur les résidus en vue de l'analyse de leur homoscédasticité.
# - 7°) <font color="navy">__"VIF"__<font color="darkgreen">: qui renvoie les coefficients de multicolinéarité des variables explicatives d'une régression linéaire multiple.
# - 8°) <font color="navy">__"decomposition_des_variables"__<font color="darkgreen">: qui renvoie le pourcentage de contribution de chaque variable explicative d'une régression linéaire multiple.
# 
# <font color="darkgreen">Pour chaque analyse des résidus d'un modèle de régression, nous observerons son éventuelle adéquation à la loi normale sur la droite de Henry, et son homoscédasticité sur le nuage de variance résiduelle.

# In[56]:


# 1°) Droite de Henry :
def residus_henry(MODEL, nom, couleur, ax):
    sm.qqplot(MODEL.resid, ax=ax, line="45", fit=True, color=couleur)
    ax.set_title("Droite de Henry - Résidus de %s" %nom, fontsize=18)
    ax.set_xlabel("Quantiles théoriques" ,fontsize=16), ax.set_ylabel("Quantiles observés", fontsize=16)
    
# 2°) Nuage de la variance résiduelle :
def nuage_residuel(MODEL, nom, couleur, ax):
    ax = plt.plot(MODEL.fittedvalues, MODEL.resid, ".", color=couleur, alpha=0.3)
    plt.title("Nuage de la variance résiduelle de %s" %nom, fontsize=18)
    plt.xlabel("Population", fontsize=16), plt.ylabel("Résidus", fontsize=16)
    
# 3°) Les deux en même temps :
def analyse_des_residus(MODEL, nom, couleur):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    residus_henry(MODEL, nom, couleur, ax1)
    nuage_residuel(MODEL, nom, couleur, ax2)
    
# 4°) Test de Shapiro :
def test_shapiro(MODEL):
    SH = shapiro(MODEL.resid)
    print("TEST DE SHAPIRO:", " statistic =", SH[0], " pvalue =", SH[1])

# 5°) Test de Kolmogorov-Smirnov :
def test_residus_KS(MODEL):
    MR = MODEL.resid
    KS = ks_2samp(MR,list(np.random.normal(MR.mean(), MR.std(), len(MR))))
    print("TEST DE KOLMOGOROV-SMIRNOV:", " statistic =", KS[0], " pvalue =", KS[1])
    
# 6°) Test de Breusch-Pagan :
def test_residus_BP(MODEL):
    BP = sm.stats.diagnostic.het_breuschpagan(MODEL.resid, MODEL.model.exog)
    print("TEST DE BREUSCH-PAGAN:", " statistic =", BP[0], " pvalue =", BP[1])

# 7°) Coefficients de multicolinéarité :
def VIF(RLM, variables):
    MEX = RLM.model.exog
    return(pd.DataFrame({"VIF de la variable": [variance_inflation_factor(MEX, i) for i in np.arange(1, MEX.shape[1])]},
                        index = variables))

# 8°) Pourcentage de contribution des variables explicatives :
def decomposition_des_variables(RLM):
    data = sm.stats.anova_lm(RLM, typ=1)
    ssq = data[["sum_sq"]]
    ssq = np.round(100 * ssq / ssq.sum(), 2)
    ssq["sum_sq"] = ssq.sum_sq
    ssq.columns = ["Pourcentage de l'explication de la variance"]
    return(ssq)


# ## <a id="m222"><font color="forestgreen" style="margin-left:60px"><I><U>II.2. Variance de la sous-nutrition</U></I></font></a>

# <font color="darkgreen">Réalisons maintenant l'ANOVA sur la variable __undernourished__, le nombre d'êtres humains en sous-nutrition. Nous analyserons également l'adéquation des résidus à la loi normale ainsi que l'homoscédasticité du modèle respectivement sur la droite de Henry et sur le nuage de la variancce résiduelle.

# In[57]:


RL1 = ols("undernourished ~ population+pib+potentiel+domestic_supply", data = world_num).fit(alpha=nt)
print(RL1.summary().tables[0]) ; analyse_des_residus(RL1, "RL1", "chocolate")


# <font color="darkgreen">Le modèle semble fiable à la vue de la normalité sensible des résidus (à un outlier près) et à leur concentration sur le nuage résiduel. Nous observons un __R² de 82,6%__, donc nous pouvons nous fier à ce modèle pour expliquer en grande partie la variance de la sous-nutrition dans le monde. Procédons maintenant à la décomposition des variables explicatives de ce modèle.

# In[58]:


decomposition_des_variables(RL1)


# <font color="darkgreen">C'est donc sans surprise que la taille de la population influe fortement sur le nombre d'êtres humains en sous-nutrition, c'est le facteur explicatif principal de cette variance. Observons ensuite le rôle du PIB par habitant et du potentiel alimentaire, tous deux corrélés entre eux. Ces deux variables n'expliquent qu'à elles seules que moins de 50% de la variance. Nous pouvons en déduire qu'un pays riche et peuple n'est pas à l'abri d'un fort taux d'êtres humains en sous-nutrition, comme  nous le verrons notamment plus tard avec l'Inde et la Chine (que nous avons pourtant retiré de cette analyse).

# ## <a id="m223"><font color="forestgreen" style="margin-left:60px"><I><U>II.3. Variance du pourcentage de la sous-nutrition</U></I></font></a>

# <font color="darkgreen">Nous allons maintenant nous intéresser au pourcentage de la population locale en sous-nutrition dans chaque pays. Nous pouvons à nouveau considérer l'Inde et la Chine. Comme pour le schéma précédent, nous interpreterons également l'analyse des résidus directement sous le sommaire du modèle.

# In[59]:


RL2 = ols("und_percentage ~ pib_hbt+potentiel", data = world_per).fit(alpha=nt)
print(RL2.summary().tables[0]) ; analyse_des_residus(RL2, "RL2", "darkorange")


# <font color="darkgreen">Pour les mêmes  raisons que dans le premier modèle, les résidus manifestent un modèle fiable. Nous parvenons à expliquer __61%__ de la variance du pourcentage de la sous-nutrition d'après le R². Décomposons donc les variables explicatives de ce modèle.

# In[60]:


decomposition_des_variables(RL2)


# <font color="darkgreen">C'est donc le potentiel alimentaire qui joue avant tout sur le pourcentage de la sous-nutrition. Cette variable s'accompagne bien sûr de la disponibilité intérieure en nourriture, en calories et en protéines comme nous pouviens l'observer sur la matrice des corrélations. C'est donc avant toute chose cette mauvaise répartition alimentaire mondiale qui creuse ce pourcentage dans beaucoup de pays.

# <div style="text-align:right"><a href="#sommaire"><I>Retour au sommaire</I></a></div>

# <BR>
# <BR>
# <BR>
# 
# # <div style="border: 10px ridge darkgoldenrod; padding:15px; margin:5px"><a id="m3"><font color="darkgoldenrod"><center>Mission 3 : Ciblage des zones dans le besoin alimentaire</center></font></a></div>

# 
# 
# # <a id="m31"><font color="darkgoldenrod"><U>I. Classification hiérarchique des états</U></font></a>

# ## <a id="m311"><font color="goldenrod" style="margin-left:60px"><U><I>I.1. Dendrogramme</I></U></font></a>

# <font color="saddlebrown">Pour cette mission, nous allons classifier chaque pays selon les six variables suivantes:
# - <font color="steelblue"><I>population</I></font> : Le nombre d'habitants.
# - <font color="steelblue"><I>und_percentage</I></font> :  Le pourcentage de la population en sous-nutrition.
# - <font color="steelblue"><I>potentiel</I></font> : Le pourcentage de la population pouvant potentiellement être nourrie avec la disponibilité intérieure locale.
# - <font color="steelblue"><I>pib_hbt</I></font> : Le PIB par habitants.
# - <font color="steelblue"><I>food_supply_kcal</I></font> : La disponibilité intérieure totale en calories.
# - <font color="steelblue"><I>food_supply_kgproteine</I></font> :  La disponibilité intérieure totale en protéines.
# 
# Nous appellerons <font color="darkorange"><B>df</B></font> la restriction de <font color="red"><B>world</B></font> ne contenant que ces 6 variables.

# In[61]:


df = world[["population", "undernourished", "und_percentage", "potentiel", "pib_hbt", 
            "food_supply_kcal", "food_supply_kgproteine"]]
df.head(100).tail(4)


# <font color="saddlebrown">Nous allons maintenant donner le tableau centré-réduit de df, que nous appellerons X_scaled, et en tirer un dendrogramme pour estimer un nombre raisonnable de clusters à effectuer.

# In[62]:


X = df.values ; std_scaler = preprocessing.StandardScaler().fit(X)
X_scaled = std_scaler.transform(X)

Z = linkage(X_scaled, "ward") ; plot_dendrogram(Z, world.index)


# <font color="saddlebrown">Nous observons qu'un découpage raisonnable pourrait se faire en 5 groupes distincts. Nous allons donc effectuer un clustering des pays étudiés selon les cinq variables citées en fonction des données centrées-réduites.

# ## <a id="m312"><font color="goldenrod" style="margin-left:60px"><I><U>I.2. Clustering et projection des états sur le 1er plan factoriel</U></I></font></a>

# <font color="saddlebrown">Nous découpons ici le dendrogramme en 5 clusters afin d"associer chaque pays à un groupe, variable que nous ajoutons à <font color="red">__world__ <font color="saddlebrown"> ainsi qu'à <font color="darkorange">__df__ <font color="saddlebrown">.Nous projetons ici chaque pays affecté à son cluster sur le 1er plan factoriel afin de vérifier que les groupes déterminés par notre clustering sont bien distincts.

# In[63]:


clusters = fcluster(Z, 5, criterion="maxclust")
df["cluster"] = clusters ; world["cluster"] = clusters

km = KMeans(n_clusters=5) ; km.fit(X_scaled)
pca2 = decomposition.PCA(n_components=4).fit(X_scaled)
X_scaled_projected = pca2.transform(X_scaled)

dim(18,3), plt.scatter(X_scaled_projected[:, 0], X_scaled_projected[:, 1], c=clusters.astype(np.float), cmap="jet", alpha=.8)
plt.title("Projection des {} individus sur le 1er plan factoriel".format(X_scaled_projected.shape[0]),fontsize=25); plt.show()


# <font color="saddlebrown">Le clustering disperse bien les cinq clusters formés en cinq groupes distincts sur le plan factoriel, ce qui assure la fiabilité du modèle.

# ## <a id="m313"><font color="goldenrod" style="margin-left:60px"><U><I>I.3. Description des clusters</I></U></font></a>

# <font color="saddlebrown">Bien entendu, nous n'allons pas procéder à notre analyser sur les cinq groupes de pays obtenus sans d'abord les visualiser sur notre carte mondiale et affichons en-dessous la liste des pays étudiés selon le cluster déterminé par notre algorithme.

# In[64]:


palette=["limegreen", "crimson", "goldenrod", "mediumvioletred", "darkslateblue"] # On attribuera ces couleurs aux 5 clusters.

def data_cluster(data, n): return(data.loc[data["cluster"]==n])
def world_cluster(n): return(data_cluster(world, n)) # Nous renvoie la liste des pays affectés au cluster n.

for i in range(1, 6): 
    vars()["w"+str(i)] = world_cluster(i) # wi pour i allant de 1 à 5 sera la restriction de world au cluster i.
    vars()["C"+str(i)] = mapzone(vars()["w"+str(i)]) # Liste des codes ISO des pays du cluster i.

dim(22, 7) ; ax = carte(plt.axes(projection=ccrs.PlateCarree()))
cartography([C1, C2, C3, C4, C5], palette, ax)

plt.title("Clustering Alimentaire Mondial", fontsize=40, pad=10)
plt.legend(legendes(palette), ["Cluster %s" %i for i in range(1, 6)], loc="lower left", fontsize=20) ; plt.show()

cl=pd.DataFrame({"Cluster %s" %i: [0]*55 for i in range(1, 6)})
for i in range(1, 6):
    CLI="Cluster %i" %i
    countries=list(df.loc[df["cluster"]==i].index.values)
    cl[CLI]=countries+["-"]*(55-len(countries))    
cl


# <font color="saddlebrown">Il est maintenant intéressant d'étudier ce que représentent ces cinq clusters si l'on étudie les variables deux à deux, et relever ce qui est significatif pour chacun de ces groupes.

# In[38]:


sns.pairplot(df, vars=df.drop(columns="cluster").columns, palette=palette, hue="cluster", diag_kws={"bw": 0.1}); plt.show()


# <font color="saddlebrown">Nous nous aiderons de ces graphiques pour définir au mieux les clusters. Remarquons au passage que le clustering se démarque bien sur chaque graphe. On affiche maintenant les valeurs moyennes des variables de chacun de ces clusters.

# In[65]:


groupes = pd.DataFrame(df.groupby("cluster").mean())
groupes["Nombre de pays"] = [len(df.loc[df["cluster"]==i]) for i in range(1, 6)]  ;  groupes


# <div style="border:1px solid black; padding:5px"><font color="darkgreen"><B>Cluster 1:</B> Il ne contient que l'Inde et la Chine. Outliers dans nos données de par leur population excessivement grande par rapport aux autres. Ce sont deux pays puissants économiquement mais dont la démographie tend à la sous-nutrition. Comme présenté en introduction, il s'agit des deux pays recenssant la plus forte sous-nutrition en terme de nombre d'habitants, mais pas en terme de pourcentage de la population locale.</font></div>
# 
# <div style="border:1px solid black; padding:5px"><font color="crimson"><B>Cluster 2:</B> Il regroupe les états comprenant les pourcentages de population en sous-nutrition les plus élevés. On y trouve par exemple la Zimbabwe et la République Centrafricaine où ces pourcentages s'élèvent respectivement à 59,71% et à 60,92%. On y observe donc:
# <ul><li>30 pays, en relevant massivement des états d'Afrique Subsaharienne et d'Asie Centrale.</li>
# <li>Un total d'êtres humains en sous-nutrition très important.</li>
# <li>Les pourcentages de la population locale en sous-nutrition dans chaque pays sont très élevés.</li>
# <li>Potentiel nutritif insuffisant.</li>
# <li>PIB par habitant faible.</li>
# <li>Disponibilité alimentaire intérieure faible.</li></ul></font></div>
# 
# <div style="border:1px solid black; padding:5px"><font color="darkorange"><B>Cluster 3:</B> Il regroupe des pays développés, notamment d'Europe de l'ouest et d'Amérique du nord. Ce sont les pays où le PIB par habitant est le plus fort, ainsi que les moins touchés par la sous-nutrition. On y observe donc:
# <ul><li>27 pays développés, notamment des pays de l'Union Européenne, d'Europe centrale ainsi que les États-Unis.</li>
# <li>Un taux d'êtres humains en sous-nutrition très faible.</li>
# <li>Un PIB par habitant très élevé et une disponibilité alimentaire excédentaire.</li></ul></font></div>
# 
# <div style="border:1px solid black; padding:5px"><font color="purple"><B>Cluster 4:</B> Ces pays se distinguent du cluster 2 par leur potentiel à nourrir leur population. Ils présentent également un pourcentage significatif de leur population en sous-nutrition, mais dont la disponibilité intérieure est très élevée et où le potentiel nutritif du pays est suffisant (supérieur à 100%). Nous pourrons donc les établir comme les pays en sous-nutrition dont la famine est liée à des problèmes internes. On y observe donc:
# <ul><li>55 pays, dont des états d'Afrique Centrale, d'Asie Occidentale et d'Amérique Latine.</li>
# <li>Pays à faible population, disponibilité alimentaire basse à l'échelle mondiale.</li>
# <li>Un taux d'êtres humains en sous-nutrition sensiblement élevé et des pourcentages locaux plutôt faibles.</li>
# <li>PIB par habitant relativement faible.</li>
# <li>Potentiel nutritif fragile.</li></ul></font></div>
# 
# <div style="border:1px solid black; padding:5px"><font color="darkslateblue"><B>Cluster 5:</B> Ce dernier cluster, semblable au cluster 3, se distingue de ce dernier par un taux de sous-nutrition bien présent. On peut donc présumer que la famine relate de problèmes d'ordre politique dans ces pays, plutôt que de problèmes de disponibilité alimentaire. On y observe donc:
# <ul><li>52 pays, Afrique du Nord, Europe de l'Est et Amérique du Sud.</li>
# <li>Des pays relativement développés, au potentiel alimentaire autosuffisant mais présentant tout de même la sous-nutrition.</li></ul></font></div>

# In[40]:


world.to_csv("data/projet_R/world.csv") # On enregistre notre DataFrame world mis à jour avec les clusters.


# <BR>
#     
# # <a id="m32"><font color="darkgoldenrod"><U>II. Analyses graphiques interclusters</U></font></a>

# ## <a id="m321"><font color="goldenrod" style="margin-left:60px"><I><U>II.1. Fonctions préliminaires</U></I></font></a>

# <font color="saddlebrown">Nous nous servirons tout au long de cette partie des fonctions:
# - 1°) <font color="navy">__data_cluster__<font color="saddlebrown">: qui prend en argument un DataFrame et un entier n compris entre 1 et 5 et qui renvoie la restriction de ce DataFrame au cluster n.
# - 2°) <font color="navy">__data_pie__<font color="saddlebrown">: qui prend en argument un DataFrame et une de ses variables X pour nous renvoyer un camembert donnant la répartition en pourcentage de la somme totale de la variable X prise en argument pour chaque cluster dans ce DataFrame.
# - 3°) <font color="navy">__data_bar__<font color="saddlebrown">: qui prend en argument un DataFrame et une de ces variables X  pour nous renvoyer un diagramme en tuyau d'orgues coloré selon les clusters, avec les pays en abscisses et la variable X en ordonnées.
# - 4°) <font color="navy">__data_per__<font color="saddlebrown">: qui représentera mieux le diagramme si X est une variable en pourcentage.
# - 5°) <font color="navy">__world_cluster, world_pie et world_bar/per__<font color="saddlebrown">: respectivement ces trois dernières fonctions appliquées direcement au DataFrame <font color="crimson">__world__<font color="saddlebrown">.

# In[66]:


# 1°) data_cluster:
def data_cluster(data, n): return(data.loc[data["cluster"]==n])

# 2°) data_pie:
def data_pie(data, X):
    pd.DataFrame(data[[X, "cluster"]].groupby("cluster").sum())[X].plot(kind="pie", colors=palette, labels=["","","","",""],
                                           autopct=lambda x: str(round(x, 2))+"%", shadow=True, textprops={"color":"w"})
    plt.legend(["Cluster %s" %i for i in range(1, 6)], loc="lower right", fontsize=10) ; plt.ylabel("")
    
# 3°) data_bar:
def data_bar(data, X):
    for i in range(1, 6): vars()["w"+str(i)] = data_cluster(data, i)
    for i, wx in enumerate([w1, w2, w3, w4, w5]):
        if len(wx)!=0: plt.bar(wx.index, wx[X], color=palette[i], label="Cluster {}".format(i+1))
    if max(world[X])<500 and min(world[X])!=0: plt.axhline(100, linestyle="--", color="black", linewidth=0.8)
    plt.xticks(fontsize=5, rotation=90); plt.legend(fontsize=15, loc="upper right")
    
# 4°) data_per:
def data_per(data, X, axx):
    for i in range(1, 6): vars()["w"+str(i)] = data_cluster(data, i)
    for i, wx in enumerate([w1, w2, w3, w4, w5]):
        if len(wx)!=0: axx.bar(wx.index, wx[X], color=palette[i], label="Cluster {}".format(i+1))
    if max(world[X])<500 and min(world[X])!=0: axx.axhline(100, linestyle="--", color="black", linewidth=0.8)
    axx.set_yticklabels(["{}%".format(int(x)) for x in axes.get_yticks()], fontsize=15)
    plt.xticks(fontsize=5, rotation=90); plt.legend(fontsize=15, loc="upper right")
    
# 5°) Fonctions associées directement à WORLD:
def world_cluster(n): return(data_cluster(world, n)) ;
def world_pie(X): data_pie(world, X) ;
def world_bar(X): data_bar(world, X) ;
def world_per(X, axx): data_per(world, X, axx)


# ## <a id="m322"><font color="goldenrod" style="margin-left:60px"><U><I>II.2. Représentations des clusters à l'échelle mondiale</I></U></font></a>

# <font color="saddlebrown">Commençons donc par observer la répartition de la population mondiale et la répartition du nombre d'êtres humains en sous-nutrition parmi les clusters.

# In[42]:


dim(18, 12), plt.subplot(2, 2, 1) , world_pie("population") ; plt.title("Population mondiale", fontsize=20)
plt.subplot(2, 2, 2) , world_pie("undernourished") ; plt.title("Taux de la sous-nutrition mondiale", fontsize=20); plt.show()

dim(15, 5); k=1
for X in ["domestic_supply", "food_supply_kcal", "food_supply_kgproteine"]:
    plt.subplot(1, 3, k)
    world_pie(X)
    if k==1: plt.title("Répartition de la disponibilité alimentaire mondiale")
    elif k==2: plt.title("Répartition de la disponibilité alimentaire en calories")
    else: plt.title("Répartition de la disponibilité alimentaire en protéines")
    k+=1
plt.show()


# <font color="saddlebrown">Rappelons que le cluster 1 représente l'Inde et la Chine. Ces deux pays représentent en effet près de 38% de la population mondiale, ainsi que 42% de la sous-nutrition globale. Ce sont des zones cibles prioritaires dans le cadre de la lutte contre la faim dans le monde. Nous pourrons également observer les pays du cluster 2, représentant 15,85% de la population totale ainsi que 37,3% de la sous-nutrition. Ces taux s'expliquent par des pays à population plus faibles, mais recensant donc un pourcentage de la population locale en sous-nutrition plus important. Enfin, remarquons que le cluster 4 est relativement concerné par les mêmes critères que le cluster 2, mais avec des proportions plus faibles. Observons maintenant les différentes répartitions de la disponibilité alimentaie mondiale parmi ces clusters.
#     
# <font color="saddlebrown">Remarquons que seuls les Clusters 3 et 5 possèdent une disponibilité alimentaire plus importante que leur représentation démographique à l'échelle mondiale. En effet, c'est dans ces clusters que nous trouverons les états autosuffisants voire excédentaires, mais nous recensons tout de même la sous-nutrition dans le cluster 5. Rappelant que nous avons observé lors de la mission 1 que nous pouvions aisément nourrir plus de 120% de la population, établir que 1% du monde possède 1% de la disponibilité alimentaire mondiale est alors déjà un critère d'excédentaire. L'Inde et la Chine présentent des taux similaires pour ce qui est de leur taux de population mondiale respectifs et de leurs taux de disponibilité alimentaires mondiale.
#     
# <font color="saddlebrown">Nous pouvons donc émettre les hypothèses suivantes:
# - Les pays des clusters 1 et 5 doivent leurs taux de sous-nutrition à leur politique intérieure. Ils disposent de ressources suffisantes pour nourrir leur population mais ne les distribuent pas à bon escient.
# - Les pays du cluster 3 sont généralement excédentaires. Ils sont autosuffisants et n'auront pour l'instant pas recours à un remaniement politique intérieur ou à une quelconque aide humanitaire.
# - Les pays des clusters 2 et 4 requierent, quant à eux, une aide extérieure. C'est leur manque de disponibilité alimentaire, conjugué à une population trop importante, qui est à l'origine de leurs taux de sous-nutrition.

# ## <a id="m323"><font color="goldenrod" style="margin-left:60px"><U><I>II.3. Des potentiels alimentaires irréguliers</I></U></font></a>

# <font color="saddlebrown">Nous nous intéresserons maintenant aux différents potentiels alimentaires. Le graphique suivant reprend les résultats de la partie III.2. de la mission 1 mais colorie cette fois-ci les états selon leur cluster.

# In[43]:


figure = plt.figure(figsize=(20, 6)) ; axes = figure.add_subplot(111)
world_per("potentiel", axes) ; axes.set_title("Division des potentiels alimentaires", fontsize=30)
axes.set_ylabel("Potentiel alimentaire", fontsize=20); plt.yticks(fontsize=20); plt.show()


# <font color="saddlebrown">Nous pouvons appuyer l'analyse des camemberts tracés dans la sous-partie précédente par l'observation de ce nouveau graphique: Les Clusters 3 et 5 sont clairement excédentaires, ce qui révèle donc un problème d'ordre politique concernant la sous-nutrition des pays du cluster 5 et non un manque de ressources alimentaires. C'est le contraire pour le cluster 2, où le potentiel alimentaire est insuffisant, c'est donc notamment ce manque de disponibilité alimentaire qui en est à l'origine. Le potentiel alimentaire des pays du cluster 4 est soit insuffisant, soit fragile, n'excédant pas les 120%. Les ressources y sont également insuffisante. Ce sont donc des pays dans le besoin de ressources alimentaires mais qui nécessitent également un remaniement politique afin d'équilibrer la disposition de leurs ressources. Outre l'aspect population, nous pouvons considérer la Chine comme un pays du cluster 5 (fort potentiel alimentaire mais sous-nutrition tout de même présente) et l'Inde comme un pays du Cluster 4 (potentiel alimentaire fragile et sous-nutrition présente).

# ## <a id="m324"><font color="goldenrod" style="margin-left:60px"><U><I>II.4. Inégalités économiques</I></U></font></a>

# <font color="saddlebrown">Il n'est pas surprenant de mettre en avant les inégalités économiques pour expliquer pourquoi certains pays souffrent encore aujourd'hui de sous-nutrition, et d'autres non. Ce graphique résume ces inégalités selon les clusters.

# In[44]:


dim(20, 7); world_bar("pib_hbt"), plt.title("Inégalités économiques", fontsize=25)
plt.ylabel("PIB par habitant (dollar américain)", fontsize=20) ; plt.yticks(fontsize=15) ; plt.show()


# <font color="saddlebrown">Naturellement, on relève un taux de richesse démesuré parmi les pays du cluster 3 par rapport aux autres. Remarquons de sensibles similitudes entre les clusters 4 et 5, ainsi qu'entre les clusters 1 et 2. Ce projet ayant pour but de contourner au mieux l'aspect économique, nous pouvons néanmoins mettre en avant le taux pauvreté présent en Inde et en Chine, pays surpeuplés, contribuant à la forte quantité encore actuelle d'êtres humains souffrant de la famine. Tandis que le pourcentage de la population en sous-nutrition dans certains pays s'expliquera par la pauvreté plutôt dans les pays des clusters 4 et 5.

# <BR/>
#     
# # <a id="m33"><font color="darkgoldenrod"><U>III. Observations  approfondies</U></font></a>

# ## <a id="m331"><font color="goldenrod" style="margin-left:60px"><U><I>III.1. Besoins alimentaires</I></U></font></a>

# <font color="saddlebrown">Divisons maintenant les besoins en protéines et en calories. <font color="saddlebrown">__Notons tout d'abord que seuls les clusters 2 et 4 admettent des pays dont le potentiel alimentaire est inférieur à 100%.__ C'est pourquoi nous allons, pour le moment, ne classifier les besoins en calories et en protéines que pour les pays de ces deux clusters.

# In[45]:


fig, ax = plt.subplots(figsize=(15, 6)); plt.axhline(100, color="black") ; plt.axvline(100, color="black") ; k=1
for data in [w1, w2, w3, w4, w5]:
    data.plot.scatter(ax=ax, x="potentiel_calorie", y="potentiel_proteine", s=110, color=palette[k-1], label="Cluster %s" %k)
    k+=1
plt.title("Potentiel alimentaire selon les clusters", fontsize=30)
plt.xlabel("Potentiel nutritif en calories (%)", fontsize=20) ; plt.ylabel("Potentiel nutritif en protéines (%)", fontsize=20)
plt.xticks(fontsize=15) ; plt.yticks(fontsize=15) ; plt.legend(loc="upper left", fontsize=20)
plt.xlim(70, 180) ; plt.ylim(70, 250) ; plt.show()


# <font color="saddlebrown">Remarquons tout d'abord que les pays ayant un potentiel alimentaire insuffisant pour chacune des deux unités d'énergie (calories et protéines) sont tous des pays du cluster 2. Ce cluster comprend donc bel et bien les pays dont l'aide alimentaire est la plus urgente. Les pays du cluster 4 ne souffrent que d'une seule de ces deux insuffisances, si ce n'est d'aucune à défaut d'une certaine fragilité. Pour les clusters restants, nous ne relevons pas de distinction particulière entre calories et protéines pour ce qui est de leurs éventuels besoins alimentaires.

# ## <a id="m332"><font color="goldenrod" style="margin-left:60px"><U><I>III.2. L'Inde et la Chine</I></U></font></a>

# <font color="saddlebrown">Lors du clustering, deux états se sont distingués du reste du monde de par la taille de leur population: l'Inde et la Chine. Ils forment à eux seuls le cluster 1 et nous consacrerons cette partie à l'étude des données récoltées pour ces deux pays.

# In[46]:


dim(15, 6), plt.axhline(100, linestyle="--", color="black", linewidth=1)

plt.bar(0.3, w1.loc["Chine", "potentiel_calorie"], color="darkkhaki", label="Poteniel calorique")
plt.bar(1, w1.loc["Chine", "potentiel_proteine"], width=0.7, color="dimgrey", label="Potentiel protéines")
plt.bar(1.7, 100-w1.loc["Chine", "und_percentage"], width=0.7, color="mediumseagreen", label="Population alimentée")
plt.legend(loc="upper right", fontsize=20)

plt.bar(3.3, w1.loc["Inde", "potentiel_calorie"], color="darkkhaki", label="Poteniel calorique")
plt.bar(4, w1.loc["Inde", "potentiel_proteine"], width=0.7, color="dimgrey", label="Potentiel protéines")
plt.bar(4.7, 100-w1.loc["Inde", "und_percentage"], width=0.7, color="mediumseagreen", label="Population alimentée")

plt.yticks(fontsize=18) ; plt.ylabel("Pourcentage", fontsize=18)
plt.xticks(range(0, 6), ["", "Chine", "", "", "Inde", ""], fontsize=20)
plt.title("La Chine et l'Inde, des pays autosuffisants?", fontsize=25, color="green"), plt.grid(False) ; plt.show()


# <font color="saddlebrown">La Chine doit donc clairement procéder à un meilleur équilibre de sa disponibilité alimentaire pour venir à bout de la famine au sein du pays. Nous pouvons donc même reconsidérer ce pays comme un du cluster 5, où l'aide alimentaire n'est pas forcément nécessaire mais où la politique de distribution n'est pas forcément correcte. En revanche, l'Inde ne semble pas disposer de ressources suffisantes pour remédier à elle seule au problème. On observe que nous ne parviendrions pas à une population nourrie à 100% si même l'ensemble de la disponibilité alimentaire locale était parfaitement distribuée et équitable. L'Inde requiert donc une aide extérieure très forte, et en dépit de sa population, nous devons avoir la même considération pour ce pays que pour ceux du cluster 2, à savoir un des pays prioritaires dans les besoins d'aides alimentaires extérieures.

# ## <a id="m333"><font color="goldenrod" style="margin-left:60px"><U><I>III.3. Des pays dans l'urgence alimentaire</I></U></font></a>

# <font color="saddlebrown">Nous avons observé la situation de l'Inde et de la Chine, mais observons maintenant de façons générale les taux de sous-nutrition donnés selon les clusters.

# In[47]:


figure = plt.figure(figsize=(20, 5)) ; axes = figure.add_subplot(111) ; world_per("und_percentage", axes)
axes.set_title("Comparaison des pourcentages de sous-nutrition", fontsize=30)
axes.set_ylabel("Population en sous-nutrition", fontsize=20), plt.yticks(fontsize=15); plt.show()


# <font color="saddlebrown">Le Cluster 2 se distingue des quatre autres groupes de pays avec une marge importante. Comme mentionné en mission 1, nous observons des pourcentages de sous-nutrition locale allant jusqu'à 60%. Analysons donc les besoins des pays du cluster 2 en les mentionnant.

# In[48]:


fig, ax = plt.subplots(figsize=(10, 7)); plt.axhline(100, color="black") ; plt.axvline(100, color="black")
w2.plot.scatter(ax=ax, x="potentiel_calorie", y="potentiel_proteine", s=80, color="crimson", fig=fig)
for (i, txt) in enumerate(w2.index):
    ax.annotate(txt, (w2["potentiel_calorie"][i], w2["potentiel_proteine"][i]), 
                size=13, xytext=(0,0), ha="left", textcoords="offset points")
plt.xlim(70, 130) ; plt.ylim(70, 130)
plt.xlabel("Potentiel nutritif en calories (%)", fontsize=15) ; plt.ylabel("Potentiel nutritif en protéines (%)", fontsize=15)
plt.title("Le Cluster 2: des pays au potentiel alimentaire insuffisant", fontsize=20, color="crimson", pad=10) ; plt.show()


# <font color="saddlebrown">Sont donc affichés ici les pays dans l'urgence alimentaire la plus rouge. Nous trouvons donc avant tout des pays d'Afrique et d'Asie Centrale. Distinguons la République Centrafricaine, Madagascar et le Libéria où le potentiel alimentaire est globalement on-ne-peut-plus faible. Les potentiels alimentaires au-dessus de 100% pour les deux unités d'énergie pour le cluster 2 restent tout de même très fragile, n'excédant pas tous deux 120%, ce qui est relativement faible comparé à un pays auto-suffisant comme ceux des clusters 3 et 5. Nous pourrons donc considérer l'ensemble des pays du cluster 2 comme des pays ayant recours à une aide alimentaire extérieure, humanitaire ou plan d'états international. Nous avons relevé un potentiel alimentaire fragile pour les pays du cluster 4, c'est pourquoi nous nous penchons également sur les besoins de ces pays sur le graphique suivant. 

# In[49]:


fig, ax = plt.subplots(figsize=(10, 7)); plt.axhline(100, color="black") ; plt.axvline(100, color="black")
w4.plot.scatter(ax=ax, x="potentiel_calorie", y="potentiel_proteine", s=80, color="purple", fig=fig)
for (i, txt) in enumerate(w4.index):
    ax.annotate(txt, (w4["potentiel_calorie"][i], w4["potentiel_proteine"][i]), 
                size=13, xytext=(0,0), ha="left", textcoords="offset points")
plt.xlim(85, 170) ; plt.ylim(90, 170) 
plt.xlabel("Potentiel nutritif en calories (%)", fontsize=15) ; plt.ylabel("Potentiel nutritif en protéines (%)", fontsize=15)
plt.title("Le Cluster 4: des pays au potentiel alimentaire fragile", fontsize=20, color="purple", pad=10) ; plt.show()


# <font color="saddlebrown">Nous pouvons alors porter une priorité équivalente aux pays du cluster 2, aux Bahamas, aux Îles Salomon et à l'Île de Sao-Tomé pour ce qui est de l'urgence alimentaire. Ces pays souffrent d'une insuffisance pour une unité d'énergie et d'une fragilité pour la deuxième. Les pays restants seront à prendre en compte également, dans une mesure d'urgence moindre, mais également importante.

# <div style="text-align:right"><a href="#sommaire"><I>Retour au sommaire</I></a></div>

# <BR>
# <BR>
# <BR>
# 
# # <div style="border: 10px ridge darkred; padding:15px; margin:5px"><a id="m4"><font color="darkred"><center>Mission 4 : Axes de lutte contre la faim dans le monde</center></font></a></div>

# <div style="border: 2px ridge darkred; padding:5px; text-align:justify"><B><I>Dans cette dernière mission du projet, nous ne considérerons plus à présent que le monde dispose largement de quoi nourrir le monde entier, comme en témoigne les résultats de la partie III.2. de la mission 1. Nous ne tiendrons plus compte non plus de la différenciation entre les deux unités d'énergie, calories et protéines. Nous nous baserons sur l'état actuel de la sous-nutrition dans le monde, sur les différents types d'utilisation des productions alimentaires, et en proposerons quelques idées de réaménagement en vue de "libérer" des disponibilités alimentaires supplémentaires que l'on pourrait spécifiquement destiner aux êtres humains souffrants de sous-nutrition. Nous considérerons cependant le clustering effectué en mission 3 pour globaliser certaines zones du monde.</I></B></div>

# 
# # <a id="m41"><font color="darkred"><U>I. Utilisation de la disponibilité alimentaire</U></font></a>

# ## <a id="m411"><font color="crimson" style="margin-left:60px"><I><U>I.1. Remaniement du DataFrame et présentation de nouvelles variables</U></I></font></a>

# <font color="darkred">Pour cette mission, nous allons centrer notre DataFrame <font color="midnightblue"><B>world</B></font> sur les variables de la sous-nutrition, la disponibilité alimentaire totale en milliers de tonnes de nourritures et le pourcentage d'exportations de cette disponibilité. Puis nous ajoutons donc les 4 variables suivantes:
# - <font color="midnightblue"><I>nourished</I></font>  : Le nombre d'êtres humains en autosuffisance alimentaire (population mondiale optée de la population en sous-nutrition).
# - <font color="midnightblue"><I>food</I></font> : La quantité en milliers de tonnes de la disponibilité alimentaire destinée à la nourriture pour l'homme.
# - <font color="midnightblue"><I>feed</I></font> : La quantitée en milliers de tonnes de la disponibilité alimentaire destinée aux animaux.
# - <font color="midnightblue"><I>waste</I></font> : La quantité en milliers de tonnes de la disponibilité alimentaire faisant l'objet de pertes.
# - <font color="midnightblue"><I>other_uses</I></font> : La quantité en milliers de tonnes de la disponibilité alimentaire restant, destinée à d'autres types de productions.
# - <font color="midnightblue"><I>food_per</I></font> : Le pourcentage de la disponibilité alimentaire destinée à la nourriture au sein d'un pays donné.
# - <font color="midnightblue"><I>feed_per</I></font> : Le pourcentage de la disponibilité alimentaire destinée aux animaux.
# - <font color="midnightblue"><I>waste_per</I></font> : Le pourcentage de la disponibilité alimentaire faisant l'objet de pertes.
# - <font color="midnightblue"><I>other_uses_per</I></font> : Le pourcentage de la disponibilité alimentaire restant, destinée à d'autres types de productions (cosmétie, carburant...).
# - <font color="midnightblue"><I>exportation</I></font> : Qui représentera le pourcentage de la disponibilité alimentaire exportée vers des pays extérieurs.

# In[65]:


gf=pd.read_csv("data/projet_R/ou.csv").set_index("country")
world=pd.read_csv("data/projet_R/world.csv").set_index("country")
world=world[["cluster", "population", "undernourished", "und_percentage", "potentiel", "domestic_supply"]]
world=world.join(gf) # Introduction des pourcentages d'utilisations.
world["nourished"]=world["population"]-world["undernourished"] # Taux d'êtres humains rassasiés, inverse de undernourished.

world["food"]=world["domestic_supply"]*world["food_per"]/100 # Introduction des quantités d'utilisations.
world["feed"]=world["domestic_supply"]*world["feed_per"]/100
world["waste"]=world["domestic_supply"]*world["waste_per"]/100
world["other_uses"]=world["domestic_supply"]*world["other_uses_per"]/100

exp=pd.read_csv("data/projet_R/W/alim.csv").set_index("country") # Introduction du pourcentage d'exportations.
exp=exp.drop(columns="domestic_supply") ; world=world.join(exp)
world.to_csv("data/projet_R/world_food_waste.csv")
world[["population", "nourished", "domestic_supply", "food", "feed", "waste", "other_uses", "exportation"]]


# ## <a id="m412"><font color="crimson" style="margin-left:60px"><I><U>I.2. Ajustement de constantes et définitions de fonction</U></I></font></a>

# <font color="darkred">Voyons si la disponibilité alimentaire en nourriture destinée à l'homme est proportionnelle à l'autosuffisance alimentaire. Nous retirerons les quelques pays à la population supérieure à 500 millions d'habitants pour une meilleure visibilité des résultats.

# In[51]:


dim(18, 4), plt.plot(world["nourished"], world["food"], "o", color="sienna")
plt.title("Autosuffisance alimentaire proportionnelle à la disponibilité alimentaire?", fontsize=25)
plt.xlabel("Population nourrie (en milliard d'habitants)", fontsize=16) ; plt.xticks(fontsize=20)
plt.ylabel("Quantité alimentaire disponible (en kg)", fontsize=17) ; plt.yticks(fontsize=13)
plt.xlim(0, 2*10**8) ; plt.ylim(0, 2*10**5) ; plt.show()


# <font color="darkred">On constate qu'une droite croissante se dégage de ce nuage de points par régression linéaire. On en déduit une proportionnalité sensible entre la quantité de nourriture disponible et le nombre d'êtres humains nourris. On peut donc estimer que la quantité de nourriture totale peut nourrir environ 89,54% du monde entier. Nous tirerons de cette proportion des estimations du nombre d'êtres humains supplémentaires pouvant être nourris en libérant certaines quantités de nourritures.

# In[52]:


FQ=world["food"].sum() # Disponibilité alimentaire mondiale destinée à l'homme en milliers de tonnes de nourritures.
PN=round(POPULATION-SSN,0) # Population totale en autosuffisance alimentaire.
PRCN=round(100*PN/POPULATION,2) # Pourcentage de la population totale en autosuffisance alimentaire.

print("Le monde dispose d'une quantité alimentaire totale de {} kg,".format(FQ),
      " ce qui nourrit environ {}% de la population totale,".format(PRCN)) ; print("soit {} d'êtres humains.".format(PN))


# <font color="darkred">En vue des analyses répétées que nous exécuterons, nous nous servirons de la fonction <font color="navy"><I><B>barre_nutritive</B></I><font color="darkred"> (suivez bien, cette fonction est particulière!) qui prendra notamment en argument une zone (le monde entier, un cluster ou un pays), le pourcentage de la population de cette zone en autosuffisance alimentaire A, ainsi qu'un pourcentage de cette même population B susceptible d'être nourrie par une quelconque libération alimentaire extérieure ut (pertes alimentaires ou nourritures pour animaux par exemple). Cette fonction nous renverra alors un diagramme horizontal unique, sommant la population nourrie et la population susceptible d'être nourrie par la libération alimentaire citée, et nous donnant donc une estimation du pourcentage de la population de cette zone pouvant être nourrie si l'on libérait cette dite disponibilité alimentaire. Nous donnerons par ailleurs une couleur spécifique en argument pour pouvoir distinguer les barres obtenues. L'exemple de la partie II.1. de la mission 4 qui va suivre vous permettra de comprendre l'utilisation de cette fonction si mon explication n'est pas claire.

# In[53]:


def barre_nutritive(A, B, couleur, zone, ut, s): 
    dim(18, 2)
    plt.barh(" ", A, color=couleur, label="Population du {} en autosuffisance alimentaire (en %)".format(zone))
    plt.barh(" ", B, left=A, color="darkkhaki", 
             label="Population du {} pouvant être nourrie avec la disp. alimentaire libérée par {} (en %)".format(zone, ut))
    plt.xticks(range(0, s, 5), fontsize=20)
    plt.legend(loc="upper left", fontsize=15)


# <font color="darkred">Nous nous pencherons dans cette dernière mission sur la distribution des ressources alimentaires.

# ## <a id="m413"><font color="crimson" style="margin-left:60px"><I><U>I.3. Taux d'utilisation de la disponibilité alimentaire</U></I></font></a>

# <font color="darkred">Voyons dans un premier temps comment est globalement utilisée cette disponibilité alimentaire sur un diagramme circulaire.

# In[54]:


s1, s2, s3, s4 = world["food"].sum(), world["feed"].sum(), world["other_uses"].sum(), world["waste"].sum()

dim(6, 6), plt.pie([s2, s1, s3, s4], labels = ["Nourriture pour animaux", "Nourriture", "Autres utilisations", "Pertes"],
                   colors=["seagreen", "darkkhaki", "dimgray", "crimson"], autopct=lambda x: str(round(x, 2))+"%",
                   shadow=True, explode = [0, 0, 0, 0.4])

plt.title("Utilisation globale de la disponibilité alimentaire",fontsize = 20) ; plt.ylabel(""); plt.axis("equal"); plt.show()


# <font color="darkred">Notons tout d'abord que presque la moitié de la disponibilité alimentaire mondiale n'est pas destinée à nourrir la population.<BR/>
# <B>Nous relevons sinon un taux de pertes assez important de 5,54%.</B>

# <BR/>
# 
# # <a id="m42"><font color="darkred"><U>II. Pertes alimentaires</U></font></a>

# ## <a id="m421"><font color="crimson" style="margin-left:60px"><I><U>II.1. Potentiel alimentaire pouvant être libéré</U></I></font></a>

# <font color="darkred">Imaginons maintenant que nous distribuions les pertes alimentaires de façon équitablement répartie à l'ensemble de la population mondiale en sous-nutrition. Si l'on estime du graphique précédent que <B>47,44% de la disponibilité alimentaire totale peut nourrir 89,54% de la population mondiale</B>, il suffit d'un simple calcul mathématique pour voir ce que les <B>5,54% de pertes auraient pu rassasier</B>.

# In[55]:


PT=PRCN*5.54/47.44 # Pourcentage de la population mondiale pouvant être nourrie par l'ensemble des pertes alimentaires.
PT=round(PT, 2)

barre_nutritive(PRCN, PT, "palevioletred", "monde", "pertes", 106)
plt.title("Quelle proportion du monde pourrions-nous nourrir avec les pertes alimentaires?", fontsize=25); plt.show()

print("On nourrirait donc {}% du monde si les pertes alimentaires étaient consommées équitablement par la".format(PRCN+PT))
print("population en sous-nutrition")


# <font color="darkred">Il nous faut admettre que ce score rond de 100%, fruit de notre estimation, est un hasard. En considérant donc la qualité des paramètres de notre estimation, on peut donc admettre que les pertes libererait de justesse le monde de la faim! Nous allons maintenant repérer les zones géographiques où l'on peut relever des pertes importantes.

# ## <a id="m422"><font color="crimson" style="margin-left:60px"><I><U>II.2. Ciblage géographique</U></I></font></a>

# <font color="darkred">Repérons donc d'abord les pays dont le pourcentage des pertes alimentaires est le plus élevé selon leurs clusters définis à la mission 3. Ce premier graphique ne présentera donc pas de résultats pour ce qui est des pertes de quantités alimentaires.

# In[56]:


w1, w2, w3, w4, w5 = world_cluster(1), world_cluster(2), world_cluster(3), world_cluster(4), world_cluster(5)
figure = plt.figure(figsize=(20, 6)) ; axes = figure.add_subplot(111) ; c=0
for data in [w1, w2, w3, w4, w5]:
    axes.bar(data.index, data["food_per"]+data["feed_per"]+data["other_uses_per"], 
            label="Disponibilité alimentaire, cluster %s" %(c+1), color=palette[c])
    c+=1
axes.bar(world.index, world["waste_per"], bottom=world["food_per"]+world["feed_per"]+world["other_uses_per"], 
        label="Pertes", color="black")
axes.set_yticklabels(["{}%".format(int(x)) for x in axes.get_yticks()], fontsize=15)
plt.xticks("") ; plt.yticks(fontsize=20) ; plt.xlabel("Pays", fontsize=20)
plt.title("Taux de disponibilité alimentaire sans les pertes", fontsize=32) ; plt.legend(fontsize=15); plt.show()


# <font color="darkred">Remarquons que les pertes sont plus importantes dans les pays des clusters 2 et 4. En revanche, ces deux derniers clusters disposantes des quantités alimentaires les plus faibles, on peut supposer que leurs pertes ne sont pas les plus élevées en terme de quantités. C'est donc ce que nous allons maintenant observer sur le camembert suivant.

# In[57]:


dim(6, 6), world_pie("waste") ; plt.title("Pertes alimentaires à l'échelle mondiale", fontsize=(25)); plt.show()


# <font color="darkred">Les pertes alimentaires semblent équivalentes entre les pays pauvres et les pays riches en regardant les résultats pour les clusters 2, 3 et 4. En clair, la quantité alimentaire des pertes ne semble pas être corrélée au PIB du pays. La Chine et l'Inde sont à l'origine d'environ 1/3 des pertes alimentaires alors qu'ils présentent environ 42,89% de la sous-nutrition mondiale. Nous observerons donc ce que ces pertes pourraient combler au sein de ce cluster. L'observation de ce graphique nous poussera également à nous concentrer sur le cluster 5. Ce graphique montre que plus du tiers de l'ensemble des pertes alimentaires proviennent du cluster 5, alors que ce groupe de pays représentent 22,64% de la population mondiale.<BR/><BR/>
# De ces chiffres, nous pouvons illustrer une cartographie des pertes alimentaires.

# In[78]:


def Wp(a, b): return(mapzone( world.loc[(world["waste_per"]>=a) & (world["waste_per"]<b)] ))

WP1, WP2, WP3, WP4, WP5, WP6, WP7 = Wp(0, 3), Wp(3, 5), Wp(5, 10), Wp(10, 15), Sn(15, 20), Sn(20, 25), Sn(25, 100)
palette_w = [ "palevioletred", "deeppink", "mediumvioletred", "crimson", "darkred", "dimgrey","black"]

dim(22, 7); ax = plt.axes(projection=ccrs.PlateCarree()) ; ax = carte(ax)
cartography([WP1, WP2, WP3, WP4, WP5, WP6, WP7], palette_w, ax)
plt.legend(legendes(palette_w), ["Pertes alimentaires inférieures à 3%"] + ["Entre 3% et 5%"] +
                              ["Entre {0}% et {1}%".format(i, i+5) for i in range(5, 26, 5)] +
                              ["Pertes alimentaires supérieures à 25%"], loc="lower left", fontsize=13)
plt.title("Cartographie des pertes alimentaires", fontsize=40, pad=10) ; plt.show()


# <font color="darkred">On voit alors que les pires pourcentages de pertes alimentaires se concentrent dans les pays en sous-nutrition. Mais des pourcentages de pertes en 3% et 10% dans des pays du Nord comme en Europe ou aux États-Unis révèlent aussi de forts taux de pertes en terme de quantité.

# ## <a id="m423"><font color="crimson" style="margin-left:60px"><I><U>II.3. Des pertes à l'origine de la sous-nutrition locale dans certains pays</U></I></font></a>

# <font color="darkred">Nous allons maintenant regarder la proportion de la population que l'on pourrait nourrir dans les pays des clusters 1 et 5 si les pertes étaient réutilisées pour de la nourriture et équitablement distribuées aux habitants en sous-nutrition.

# In[ ]:


for i in [1, 5]:
    up1=100*w1["nourished"].sum()/w1["population"].sum()
np1=33.54*PRCN/100
barre_nutritive(up1, np1, "limegreen", "cluster 1", "pertes", 131)
plt.title("Quel pourcentage de la population du Cluster 1 pourrions-nous nourrir avec les pertes?", fontsize=25); plt.show()
    


# In[58]:


# Cluster 1:
up1=100*w1["nourished"].sum()/w1["population"].sum()
np1=33.54*PRCN/100
barre_nutritive(up1, np1, "limegreen", "cluster 1", "pertes", 131)
plt.title("Quel pourcentage de la population du Cluster 1 pourrions-nous nourrir avec les pertes?", fontsize=25); plt.show()

print("_"*120)

# Cluster 5:
up5=100*w5["nourished"].sum()/w5["population"].sum()
np5=34.23*PRCN/100
barre_nutritive(up5, np5, "darkslateblue", "cluster 5", "pertes", 131)
plt.title("Quel pourcentage de la population du Cluster 5 pourrions-nous nourrir avec les pertes?", fontsize=25); plt.show()


# <font color="darkred">En dépit de sa population exhorbitante, on peut considérer les résultats comme fragiles pour le cluster 1. En revanche, les pertes locales pourraient combler l'Inde et la Chine de la famine si correctement distribuées. Ce mouvement paraissant difficilement réalisable à bien pour ces deux pays, on peut admettre que la gestion des pertes alimentaires contribueraient à venir à bout de la famine mais ne suffirait pas. C'est pourquoi nous maintenons notre conclusion à la mission 3, stipulant qu'une aide extérieure serait nécessaire pour le Cluster 1.

# <font color="darkred">Nous avons vu que les pays du Cluster 5 disposaient tous d'un potentiel alimentaire autosuffisant (cf. mission 3), et nous avons supposé que la sous-nutrition (assez faible mais présente) était liée aux conditions politiques de ces pays pour ce qui est de la gestion alimentaire. Ces résultats nous confirment cette supposition. Sans les pertes, le cluster 5 serait nourrit à plus de 120%. Ce sera donc bien les politiques de gestion alimentaire qui seront à revoir au sein de ces pays pour y venir à bout de la sous-nutrition.

# <BR/>
# 
# # <a id="m43"><font color="darkred"><U>III. Autres libérations alimentaires</U></font></a>

# ## <a id="m431"><font color="crimson" style="margin-left:60px"><I><U>III.1. Nourriture pour animaux</U></I></font></a>

# <font color="darkred">Bien entendu, nous n'allons pas arrêter de nourrir les animaux (ce serait horrible...), mais à l'instar de la production de nourriture pour l'homme, la gestion de la production de nourriture pour l'animal peut se discuter. Prenons, par exemple, les États-Unis, et concentrons-nous sur les quantités produites, cette fois-ci, et comparons-les aux quantités alimentaires produites pour l'homme au Brésil.

# In[59]:


usa=world.loc[world.index=="États-Unis"] ; bre=world.loc[world.index=="Brésil"]
P=["États-Unis", "Brésil"] ; couleurs=["olive", "darkkhaki"] ; k=0

dim(20, 8), plt.axvline(150000, linestyle="--", color="black")
for data in (usa, bre):
    Q=P[k]
    a, b = data["food"].values[0], data["feed"].values[0]
    c, d = data["other_uses"].values[0], data["waste"].values[0]
    L=[b, a] ; U=["%s - Nourriture pour animaux" %Q, "%s Nourriture" %Q]
    plt.barh([U[k], "%s - Autres Utilisations" %Q, "%s - Pertes" %Q],
             [L[k], c, d], color=couleurs[k])
    plt.barh(U[k], L[k], color="crimson")
    plt.yticks(fontsize=15) ; plt.xticks(fontsize=15) ; k+=1
plt.title("Quantités d'utilisations de la disponibilité alimentaire - États-Unis & Brésil", fontsize=25); plt.show()


# <font color="darkred">La quantité de nourriture pour animaux produite par les États-Unis est d'environ 150000 kilogrammes, ce qui représente à peu de choses près la quantité de nourriture destinée à l'homme. Voyons combien d'êtres humains pourrions-nous nourrir si on libérait 20% de cette quantité pour l'homme.

# In[60]:


ani=0.2*100*c/FQ ; barre_nutritive(PRCN, ani, "seagreen", "monde", "produits animaux", 106)
plt.title("Quelle proportion du monde pourrait nourrir 20% de la nourriture destinée aux animaux aux USA?", fontsize=20)
plt.show()


# <font color="darkred">Environ 1% de la planète pourrait être nourri si on réorientait 20% de cette production agricole vers de la nourriture destinée à l'homme. Maintenons cependant que la production de nourriture pour animaux est essentielle, également. Cependant, quand on voit que les USA produisent 150 millions de tonnes de nourriture pour animaux tandis qu'un pays comme l'Estwani (pays de plus d'1 million d'habitants) ne dispose que de 596 milliers de tonnes de nourriture pour êtres humains, quantité environ 250 fois moins importante, on peut se demander ce que pourrait libérer, disons, 10% de la production alimentaire mondiale destinée aux animaux. C'est donc ce que nous observons maintenant.

# In[61]:


NA1P=10*world["feed"].sum()/FQ ; barre_nutritive(PRCN, NA1P, "seagreen", "monde", "10% de la production pour animaux", 106)
plt.title("En libérant 10% de la production destinée aux animaux?", fontsize=25) ; plt.show()


# <font color="darkred">L'idée reste à creuser mais il est intéressant de constater que seulement 10% de la production agricole destinée aux animaux suffirait à libérer 2,5% de la population en sous-nutrition de façon générale.

# ## <a id="m432"><font color="crimson" style="margin-left:60px"><I><U>III.2. Autres utilisations alimentaires</U></I></font></a>

# <font color="darkred">Comme pour la partie précédente, voyons ce que 10% de ces "autres utilisations" pourrait libérer comme potentiel alimentaire.

# In[62]:


NA1P=10*world["other_uses"].sum()/FQ ; barre_nutritive(PRCN, NA1P, "dimgray", "monde", "10% des autres utilisations", 106)
plt.title("Et en libérant 10% des ces autres utilisations?", fontsize=25) ; plt.show()


# <font color="darkred">Ces résultats montrent que si l'on montait à 20% de réorientation de ces autres utilisations vers de la nourriture pour l'homme, pourrait libérer de quoi sauver le monde de la faim, également.

# ## <a id="m433"><font color="crimson" style="margin-left:60px"><I><U>III.3. Taux d'exportations</U></I></font></a>

# <font color="darkred">Enfin, nous exploiterons un dernier critère contribuant à la sous-nutrition dans les pays les plus pauvres qui est le taux d'exportations. Nous avons vu que les pays les plus riches présentaient une disponibilité alimentaire excédentaire, à l'inverse des pays dans le besoin qui, eux, ne disposent pas des ressources nécessaires pour nourrir leur population. Nous verrons cependant que dans certains pays, une réorganisation alimentaire locale permettrait de subvenir aux besoins des habitants sous-alimentés.

# In[63]:


ba=world.loc[world.index.isin(["Guinée-Bissau", "Zambie", "Ouganda", "Namibie", "Djibouti", "Zimbabwe"])]
x, y1, y2, y3= range(len(ba)), ba["und_percentage"], ba["potentiel"], ba["exportation"]

dim(20, 7), plt.barh(x, y1, height=0.2, label="Sous-nutrition", color="crimson")
plt.barh([i-0.2 for i in x], y2,  height=0.2, label="Potentiel alimentaire", color="darkgrey")
plt.barh([i+0.2 for i in x] , y3,  height=0.2, label="Pourcentage d'exportation", color="orange")

plt.xticks(range(0, 121, 5), fontsize=20) ; plt.yticks(fontsize=20) ; plt.legend(fontsize=20, loc="lower right")
plt.yticks(range(len(ba)), [i for i in ba.index])
plt.title("Pourcentage d'exportations dans les pays en sous-nutrition", fontsize=30) ; plt.tight_layout()  ; plt.show()


# <font color="darkred">Nous avons détecté des pays en sous-nutrition critique où les taux d'exportations alimentaires étaient relativement important. En effet, certains chefs d'état semblent privilégier un apport financier au détriment de la rassasion de leur peuple, souffrant pourtant de la famine. Observons que chacun des pays choisis pour mettre cet aspect en avant témoignent d'un potentiel alimentaire insuffisant, ou au mieux extrêmement fragile. Plus de 20% de la population de ces pays est en sous-nutrition, allant même au-delà des 40% pour l'Ouganda, la Zambie et la Zimbabwe. Et nous relevons des taux d'exportations de plus de 5%, allant jusqu'à 25% pour la Namibie. Il est donc important de souligner ces disproportions. Bien que l'on s'en prenne à des dispositifs commerciaux qui dépassent le cadre de ce projet, nos valeurs tendent à prioriser la sassité de l'être humain, avant toute autre oeuvre marchande ou commerciale.

# <div style="text-align:right"><a href="#sommaire"><I>Retour au sommaire</I></a></div>

# <BR>
# 
# # <a id="m44"><font color="darkred"><U>IV. Conclusion</U></font></a>

# <div style="border: 2px ridge black; padding:5px; text-align: justify"><font color="darkred"><B>La faim dans le monde est un combat perpétuel que nous menons depuis des siècles. La première victoire de l'humanité sur ce fléau est sa capacité à nourrir le monde entier, comme nous avons pu le voir et le rappeler tout au long du projet. Il s'agit maintenant d'utiliser la disponibilité alimentaire mondiale à bon escient. Nous avons vu que certains pays comme l'Algérie, la Roumanie ou encore le Mexique (Cluster 5) relevaient la sous-nutrition malgré un potentiel alimentaire auto-satisfaisant. La politique de gestion alimentaire serait donc à revoir au sein même de ces pays pour faire disparaître la famine. Les relations internationales avec les pays autosuffisants jouent également un rôle clef dans cette lutte. Les exploitations agricoles pour le compte de certaines multinationales ne sont pas suffisamment dirigées vers les peuples en sous-nutrition. Nous pouvons citer l'exemple des transactions foncières au Brésil, où les habitants se voient privés de certaines terres agricoles susceptibles de sauver une partie de la population locale. Nous avons pu voir également en fin de mission 4 une variable peu exploitée qui est le pourcentage d'exportations de la disponibilité  alimentaire, où nous avons vu que certains pays en sous-nutrition comme la Zimbabwe ou l'Ouganda se tournaient vers des bénéfices extérieurs liés à l'exportation agricole au détriment de la nurtrition de leur peuple. Enfin, nous mettrons surtout en avant dans cette conclusion la mauvaise répartition de la disponibilité alimentaire mondiale. Des pays comme le Libéria et la République Centrafricaine ne disposent pas de quoi nourrir leur peuple, nous avons relevé des potentiels alimentaires particulièrement bas dans ces deux derniers pays, pour chacune des deux unités d'énergie. Nous pouvons proposer un plan en trois étapes qui consiterait à orienter les aides alimentaires internationales et humanitaires et vers les pays relevés dans le Cluster 2 ainsi que vers l'Inde en priorité, réorganiser la politique de gestion alimentaire pour la Chine et les pays des Cluster 5 et 4, pour ensuite orienté ces mêmes aides vers le Cluster 4. Ce plan peut paraître ambitieux, mais demeure réalisable en vertue des données de dispositions alimentaires présentées dans ce projet, et pourrait - espérons-le - enfin mettre un terme à la famine aux quatre coins du monde.</B></font></div>

# In[64]:


world=pd.read_csv("data/projet_R/world.csv").set_index("country") ; world["categorie"]=np.NaN

for i in world.index:
    CLUST = world.loc[world.index==i]["cluster"].values[0]
    POTENTIEL,UNDPER = world.loc[world.index==i]["potentiel"].values[0],world.loc[world.index==i]["und_percentage"].values[0] 
    if CLUST == 2: world.loc[world.index==i, "categorie"] = "UA"
    if CLUST == 3: world.loc[world.index==i, "categorie"] = "PE"
    if CLUST == 4:
        if POTENTIEL <= 100: world.loc[world.index==i, "categorie"] = "UA"
        else : world.loc[world.index==i, "categorie"] = "PF"
    if CLUST == 5:
        if UNDPER == 0:
            if POTENTIEL >= 150: world.loc[world.index==i, "categorie"] = "PE"
            else : world.loc[world.index==i, "categorie"] = "PA"
        else:
            if UNDPER >=5: world.loc[world.index==i, "categorie"] = "PF"
            else: world.loc[world.index==i, "categorie"] = "PA"
    if UNDPER >= 30: world.loc[world.index==i, "categorie"] = "UAP"
        
world.loc[world.index=="Chine", "categorie"]="PF" ; world.loc[world.index=="Inde", "categorie"]="UAP"
palette=["#D1DB88", "darkkhaki", "slategray", "crimson", "darkred"]
def categ(CAT): return(mapzone(world.loc[world["categorie"]==CAT]))
P1, P2, P3, P4, P5 = categ("PE"), categ("PA"), categ("PF"), categ("UA"), categ("UAP")

dim(22, 12) ; ax = plt.axes(projection=ccrs.PlateCarree()) ; ax = carte(ax) ; cartography([P1, P2, P3, P4, P5], palette, ax)
ax.legend(legendes(palette), 
          ["Pays excédentaires : Premiers potentiels de redistributions alimentaires extérieures", 
           "Pays autosuffisants : Gestions alimentaires satisfaisantes",
           "Potentiels fragiles et présence de sous-nutrition : Gestions alimentaires internes à revoir",
           "Pays en situation d'urgence alimentaire : Zones d'aides humanitaires nécessaires", 
           "Pays en situation d'urgence alimentaire critique : Zones d'aides humanitaires prioritaires"],
           fontsize=24, loc="lower left", bbox_to_anchor=(0, -0.45))
plt.title("Axes de lutte contre la faim dans le monde", fontsize=45, color="crimson", pad=10) ; plt.show()

