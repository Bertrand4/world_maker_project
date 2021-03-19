import pandas as pd ; import numpy as np ; from functions import * ; import os
import matplotlib.pyplot as plt ; from matplotlib.pyplot import figure ; from matplotlib.pylab import arange, plot 
import matplotlib as mpl ; from pylab import * ; import matplotlib.patches as mpatches
from collections import Counter ; from termcolor import colored

# Outils de génération des cartes du monde colorisées:
import cartopy ; import cartopy.io.shapereader as shpreader ; import cartopy.crs as ccrs
shpfilename = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
reader = shpreader.Reader(shpfilename)

mapworld=pd.read_csv("data/projet_R/mapworld.csv")

# La fonction mapzone identifie les pays présents dans "Zone" pour renvoyer la liste des codes ISO:
def mapzone(Zone): return(list(mapworld.loc[mapworld["country"].isin(Zone.index)]["country_code"]))

def legendes(COULEURS): # Légendes des cartes.
    L=[]
    for color in COULEURS: L.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=color))
    return(L)

def carte(ax): # Cette fonction génère une carte du monde vierge.
    ax=plt.axes(projection=ccrs.PlateCarree()) ; ax.add_feature(cartopy.feature.OCEAN)
    ax.set_extent([-150, 60, -25, 60]) ; return(ax)

def colorisation(iso, country, zone, couleur, ax): # Fonction de colorisation des états selon les données qu'ils présentent.
    if iso in zone: ax.add_geometries(country.geometry, ccrs.PlateCarree(), facecolor=couleur)
        
def cartography(ZONES, COULEURS, ax): # Cette fonction affiche directement chaque zone de couleurs différentes.
    countries = reader.records()
    for country in countries:
        iso = country.attributes["ADM0_A3"]
        for i in range(len(ZONES)): colorisation(iso, country, ZONES[i], COULEURS[i], ax)
