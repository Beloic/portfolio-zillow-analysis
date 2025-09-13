"""
Module de traitement des données Zillow Home Value Index
Auteur: Loic Bernard
Date: Décembre 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ZillowDataProcessor:
    """
    Classe pour traiter et nettoyer les données Zillow Home Value Index
    """
    
    def __init__(self, data_path: str):
        """
        Initialise le processeur avec le chemin vers les données
        
        Args:
            data_path (str): Chemin vers le fichier CSV
        """
        self.data_path = data_path
        self.df = None
        self.price_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV
        
        Returns:
            pd.DataFrame: Dataset chargé
        """
        print("🔄 Chargement des données...")
        self.df = pd.read_csv(self.data_path)
        
        # Identification des colonnes de prix (dates)
        self.price_columns = [col for col in self.df.columns if '-' in col and len(col) == 10]
        
        print(f"✅ Données chargées: {self.df.shape[0]} régions, {len(self.price_columns)} périodes")
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Retourne des informations sur le dataset
        
        Returns:
            dict: Dictionnaire contenant les informations
        """
        if self.df is None:
            raise ValueError("Les données n'ont pas été chargées. Utilisez load_data() d'abord.")
        
        info = {
            'dimensions': self.df.shape,
            'periodes': len(self.price_columns),
            'premiere_date': self.price_columns[0] if self.price_columns else None,
            'derniere_date': self.price_columns[-1] if self.price_columns else None,
            'regions_uniques': {
                'etats': self.df['State'].nunique(),
                'villes': self.df['City'].nunique(),
                'metropoles': self.df['Metro'].nunique(),
                'comtes': self.df['CountyName'].nunique()
            },
            'types_regions': self.df['RegionType'].value_counts().to_dict()
        }
        
        return info
    
    def clean_data(self, 
                   fill_method: str = 'forward',
                   min_non_null_ratio: float = 0.5) -> pd.DataFrame:
        """
        Nettoie les données en gérant les valeurs manquantes
        
        Args:
            fill_method (str): Méthode de remplissage ('forward', 'backward', 'interpolate')
            min_non_null_ratio (float): Ratio minimum de valeurs non-nulles pour conserver une région
            
        Returns:
            pd.DataFrame: Dataset nettoyé
        """
        if self.df is None:
            raise ValueError("Les données n'ont pas été chargées.")
        
        print("🧹 Nettoyage des données...")
        
        # Copie du dataframe pour éviter les modifications
        df_clean = self.df.copy()
        
        # Suppression des régions avec trop de valeurs manquantes
        null_ratios = df_clean[self.price_columns].isnull().sum(axis=1) / len(self.price_columns)
        mask_to_keep = null_ratios <= (1 - min_non_null_ratio)
        df_clean = df_clean[mask_to_keep]
        
        print(f"📊 Régions conservées: {len(df_clean)}/{len(self.df)} "
              f"({len(df_clean)/len(self.df)*100:.1f}%)")
        
        # Remplissage des valeurs manquantes
        if fill_method == 'forward':
            df_clean[self.price_columns] = df_clean[self.price_columns].fillna(method='ffill', axis=1)
        elif fill_method == 'backward':
            df_clean[self.price_columns] = df_clean[self.price_columns].fillna(method='bfill', axis=1)
        elif fill_method == 'interpolate':
            df_clean[self.price_columns] = df_clean[self.price_columns].interpolate(axis=1)
        
        # Suppression des lignes avec encore des valeurs manquantes
        df_clean = df_clean.dropna(subset=self.price_columns)
        
        print(f"✅ Nettoyage terminé: {len(df_clean)} régions finales")
        return df_clean
    
    def calculate_growth_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des métriques de croissance pour chaque région
        
        Args:
            df (pd.DataFrame): Dataset nettoyé
            
        Returns:
            pd.DataFrame: Dataset avec métriques ajoutées
        """
        print("📈 Calcul des métriques de croissance...")
        
        df_metrics = df.copy()
        
        # Prix initial et final
        df_metrics['prix_initial'] = df_metrics[self.price_columns[0]]
        df_metrics['prix_final'] = df_metrics[self.price_columns[-1]]
        
        # Croissance totale
        df_metrics['croissance_totale'] = (
            (df_metrics['prix_final'] - df_metrics['prix_initial']) / 
            df_metrics['prix_initial'] * 100
        )
        
        # Croissance annuelle moyenne
        years_span = len(self.price_columns) / 12  # Approximation
        df_metrics['croissance_annuelle'] = (
            (df_metrics['prix_final'] / df_metrics['prix_initial']) ** (1/years_span) - 1
        ) * 100
        
        # Volatilité (écart-type des variations mensuelles)
        price_changes = df_metrics[self.price_columns].pct_change(axis=1) * 100
        df_metrics['volatilite'] = price_changes.std(axis=1)
        
        # Prix maximum et minimum
        df_metrics['prix_max'] = df_metrics[self.price_columns].max(axis=1)
        df_metrics['prix_min'] = df_metrics[self.price_columns].min(axis=1)
        
        # Drawdown maximum
        rolling_max = df_metrics[self.price_columns].expanding(axis=1).max()
        drawdowns = (df_metrics[self.price_columns] - rolling_max) / rolling_max * 100
        df_metrics['drawdown_max'] = drawdowns.min(axis=1)
        
        print("✅ Métriques calculées")
        return df_metrics
    
    def get_top_performers(self, 
                          df: pd.DataFrame, 
                          metric: str = 'croissance_totale',
                          n: int = 10) -> pd.DataFrame:
        """
        Retourne les meilleures régions selon une métrique
        
        Args:
            df (pd.DataFrame): Dataset avec métriques
            metric (str): Métrique à utiliser
            n (int): Nombre de régions à retourner
            
        Returns:
            pd.DataFrame: Top N régions
        """
        top_regions = df.nlargest(n, metric)[
            ['RegionName', 'State', 'City', 'Metro', 'prix_initial', 'prix_final', metric]
        ].copy()
        
        return top_regions
    
    def get_regional_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée un résumé par région géographique
        
        Args:
            df (pd.DataFrame): Dataset avec métriques
            
        Returns:
            pd.DataFrame: Résumé par région
        """
        print("🗺️ Création du résumé régional...")
        
        regional_summary = df.groupby(['State', 'RegionType']).agg({
            'prix_initial': 'mean',
            'prix_final': 'mean',
            'croissance_totale': 'mean',
            'croissance_annuelle': 'mean',
            'volatilite': 'mean',
            'RegionName': 'count'
        }).round(2)
        
        regional_summary.columns = [
            'prix_moyen_initial', 'prix_moyen_final', 'croissance_moyenne',
            'croissance_annuelle_moyenne', 'volatilite_moyenne', 'nombre_regions'
        ]
        
        regional_summary = regional_summary.reset_index()
        
        print("✅ Résumé régional créé")
        return regional_summary


def main():
    """
    Fonction principale pour tester le module
    """
    # Initialisation du processeur
    processor = ZillowDataProcessor('../data/Zillow_Home_Value_Index.csv')
    
    # Chargement des données
    df = processor.load_data()
    
    # Affichage des informations
    info = processor.get_data_info()
    print("\n📋 Informations sur le dataset:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Nettoyage des données
    df_clean = processor.clean_data()
    
    # Calcul des métriques
    df_metrics = processor.calculate_growth_metrics(df_clean)
    
    # Top performers
    top_growth = processor.get_top_performers(df_metrics, 'croissance_totale', 5)
    print("\n🏆 Top 5 des régions avec la plus forte croissance:")
    print(top_growth)
    
    # Résumé régional
    regional_summary = processor.get_regional_summary(df_metrics)
    print("\n🗺️ Résumé par état et type de région:")
    print(regional_summary.head(10))


if __name__ == "__main__":
    main()
