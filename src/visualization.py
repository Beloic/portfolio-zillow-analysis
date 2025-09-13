"""
Module de visualisation pour les données Zillow Home Value Index
Auteur: Loic Bernard
Date: Décembre 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ZillowVisualizer:
    """
    Classe pour créer des visualisations des données Zillow
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialise le visualiseur
        
        Args:
            figsize (Tuple[int, int]): Taille par défaut des figures
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_price_distribution(self, 
                               df: pd.DataFrame, 
                               price_columns: List[str],
                               sample_dates: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Affiche la distribution des prix pour différentes périodes
        
        Args:
            df (pd.DataFrame): Dataset
            price_columns (List[str]): Colonnes de prix
            sample_dates (Optional[List[str]]): Dates à visualiser
            save_path (Optional[str]): Chemin pour sauvegarder
        """
        if sample_dates is None:
            # Sélection automatique de quelques dates
            n_samples = min(6, len(price_columns))
            indices = np.linspace(0, len(price_columns)-1, n_samples, dtype=int)
            sample_dates = [price_columns[i] for i in indices]
        
        n_plots = len(sample_dates)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, date_col in enumerate(sample_dates):
            if i < len(axes):
                data = df[date_col].dropna()
                axes[i].hist(data, bins=50, alpha=0.7, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'Distribution des Prix - {date_col}', fontweight='bold')
                axes[i].set_xlabel('Prix ($)')
                axes[i].set_ylabel('Fréquence')
                axes[i].grid(True, alpha=0.3)
        
        # Supprimer les subplots non utilisés
        for i in range(len(sample_dates), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regional_comparison(self, 
                                df: pd.DataFrame,
                                group_by: str = 'State',
                                metric: str = 'prix_final',
                                top_n: int = 15,
                                save_path: Optional[str] = None) -> None:
        """
        Compare les régions selon une métrique
        
        Args:
            df (pd.DataFrame): Dataset
            group_by (str): Colonne pour grouper
            metric (str): Métrique à comparer
            top_n (int): Nombre de régions à afficher
            save_path (Optional[str]): Chemin pour sauvegarder
        """
        # Calcul des moyennes par groupe
        regional_avg = df.groupby(group_by)[metric].mean().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(range(len(regional_avg)), regional_avg.values, 
                      color=self.colors[:len(regional_avg)])
        
        plt.title(f'Top {top_n} des {group_by}s par {metric}', fontsize=16, fontweight='bold')
        plt.xlabel(group_by)
        plt.ylabel(f'{metric} ($)')
        plt.xticks(range(len(regional_avg)), regional_avg.index, rotation=45)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_growth_analysis(self, 
                            df: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """
        Analyse la croissance des prix
        
        Args:
            df (pd.DataFrame): Dataset avec métriques de croissance
            save_path (Optional[str]): Chemin pour sauvegarder
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribution de la croissance totale
        axes[0, 0].hist(df['croissance_totale'].dropna(), bins=50, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Distribution de la Croissance Totale', fontweight='bold')
        axes[0, 0].set_xlabel('Croissance (%)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Croissance vs Volatilité
        scatter = axes[0, 1].scatter(df['volatilite'], df['croissance_totale'], 
                                   alpha=0.6, c=df['prix_final'], cmap='viridis')
        axes[0, 1].set_title('Croissance vs Volatilité', fontweight='bold')
        axes[0, 1].set_xlabel('Volatilité (%)')
        axes[0, 1].set_ylabel('Croissance Totale (%)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Prix Final ($)')
        
        # 3. Top 10 croissance par état
        state_growth = df.groupby('State')['croissance_totale'].mean().sort_values(ascending=False).head(10)
        axes[1, 0].barh(range(len(state_growth)), state_growth.values, color=self.colors[2])
        axes[1, 0].set_title('Top 10 États par Croissance Moyenne', fontweight='bold')
        axes[1, 0].set_xlabel('Croissance Moyenne (%)')
        axes[1, 0].set_yticks(range(len(state_growth)))
        axes[1, 0].set_yticklabels(state_growth.index)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Distribution des prix initiaux vs finaux
        axes[1, 1].scatter(df['prix_initial'], df['prix_final'], alpha=0.6, color=self.colors[3])
        axes[1, 1].plot([df['prix_initial'].min(), df['prix_initial'].max()], 
                       [df['prix_initial'].min(), df['prix_initial'].max()], 
                       'r--', alpha=0.8, label='Croissance nulle')
        axes[1, 1].set_title('Prix Initial vs Prix Final', fontweight='bold')
        axes[1, 1].set_xlabel('Prix Initial ($)')
        axes[1, 1].set_ylabel('Prix Final ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_map(self, 
                              df: pd.DataFrame,
                              metric: str = 'prix_final',
                              title: str = 'Prix Immobiliers par État') -> go.Figure:
        """
        Crée une carte interactive des prix par état
        
        Args:
            df (pd.DataFrame): Dataset
            metric (str): Métrique à afficher
            title (str): Titre de la carte
            
        Returns:
            go.Figure: Figure Plotly interactive
        """
        # Calcul des moyennes par état
        state_data = df.groupby('State')[metric].mean().reset_index()
        
        # Création de la carte
        fig = px.choropleth(
            state_data,
            locations='State',
            locationmode='USA-states',
            color=metric,
            color_continuous_scale='viridis',
            title=title,
            labels={metric: f'{metric} ($)'}
        )
        
        fig.update_layout(
            geo=dict(
                scope='usa',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            ),
            height=600,
            title_font_size=16
        )
        
        return fig
    
    def plot_time_series_sample(self, 
                               df: pd.DataFrame,
                               price_columns: List[str],
                               sample_regions: Optional[List[str]] = None,
                               n_samples: int = 5,
                               save_path: Optional[str] = None) -> None:
        """
        Affiche l'évolution temporelle pour un échantillon de régions
        
        Args:
            df (pd.DataFrame): Dataset
            price_columns (List[str]): Colonnes de prix (dates)
            sample_regions (Optional[List[str]]): Régions à afficher
            n_samples (int): Nombre de régions si sample_regions est None
            save_path (Optional[str]): Chemin pour sauvegarder
        """
        if sample_regions is None:
            # Sélection aléatoire de régions
            sample_indices = np.random.choice(len(df), n_samples, replace=False)
            sample_regions = df.iloc[sample_indices]['RegionName'].tolist()
        
        plt.figure(figsize=self.figsize)
        
        for i, region in enumerate(sample_regions):
            region_data = df[df['RegionName'] == region]
            if len(region_data) > 0:
                prices = region_data[price_columns].iloc[0].values
                # Conversion des dates en format datetime pour l'axe x
                dates = pd.to_datetime(price_columns)
                plt.plot(dates, prices, label=f"{region} ({region_data['State'].iloc[0]})", 
                        linewidth=2, color=self.colors[i % len(self.colors)])
        
        plt.title('Évolution Temporelle des Prix - Échantillon de Régions', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Prix ($)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard_summary(self, 
                                df: pd.DataFrame,
                                price_columns: List[str],
                                save_path: Optional[str] = None) -> None:
        """
        Crée un tableau de bord résumé
        
        Args:
            df (pd.DataFrame): Dataset
            price_columns (List[str]): Colonnes de prix
            save_path (Optional[str]): Chemin pour sauvegarder
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Configuration de la grille
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Distribution des prix (dernière période)
        ax1 = fig.add_subplot(gs[0, 0])
        latest_prices = df[price_columns[-1]].dropna()
        ax1.hist(latest_prices, bins=50, alpha=0.7, color=self.colors[0])
        ax1.set_title('Distribution des Prix (Dernière Période)', fontweight='bold')
        ax1.set_xlabel('Prix ($)')
        ax1.set_ylabel('Fréquence')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top 10 états par prix moyen
        ax2 = fig.add_subplot(gs[0, 1])
        state_avg = df.groupby('State')[price_columns[-1]].mean().sort_values(ascending=False).head(10)
        ax2.barh(range(len(state_avg)), state_avg.values, color=self.colors[1])
        ax2.set_title('Top 10 États par Prix Moyen', fontweight='bold')
        ax2.set_xlabel('Prix Moyen ($)')
        ax2.set_yticks(range(len(state_avg)))
        ax2.set_yticklabels(state_avg.index)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Évolution temporelle moyenne
        ax3 = fig.add_subplot(gs[0, 2:])
        avg_prices = df[price_columns].mean()
        dates = pd.to_datetime(price_columns)
        ax3.plot(dates, avg_prices, linewidth=3, color=self.colors[2])
        ax3.set_title('Évolution Temporelle du Prix Moyen', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Prix Moyen ($)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Statistiques générales
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.axis('off')
        
        stats_text = f"""
        📊 STATISTIQUES GÉNÉRALES
        
        • Nombre de régions: {len(df):,}
        • Période couverte: {price_columns[0]} à {price_columns[-1]}
        • Prix moyen (début): ${df[price_columns[0]].mean():,.0f}
        • Prix moyen (fin): ${df[price_columns[-1]].mean():,.0f}
        • Croissance moyenne: {((df[price_columns[-1]].mean() / df[price_columns[0]].mean()) - 1) * 100:.1f}%
        • États représentés: {df['State'].nunique()}
        • Types de régions: {df['RegionType'].nunique()}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 5. Distribution des types de régions
        ax5 = fig.add_subplot(gs[1, 2:])
        region_types = df['RegionType'].value_counts()
        ax5.pie(region_types.values, labels=region_types.index, autopct='%1.1f%%', 
               colors=self.colors[:len(region_types)])
        ax5.set_title('Distribution des Types de Régions', fontweight='bold')
        
        # 6. Corrélation prix initial vs final
        ax6 = fig.add_subplot(gs[2, :2])
        ax6.scatter(df[price_columns[0]], df[price_columns[-1]], alpha=0.6, color=self.colors[4])
        ax6.plot([df[price_columns[0]].min(), df[price_columns[0]].max()], 
                [df[price_columns[0]].min(), df[price_columns[0]].max()], 
                'r--', alpha=0.8, label='Croissance nulle')
        ax6.set_title('Corrélation Prix Initial vs Final', fontweight='bold')
        ax6.set_xlabel('Prix Initial ($)')
        ax6.set_ylabel('Prix Final ($)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Top 5 régions les plus chères
        ax7 = fig.add_subplot(gs[2, 2:])
        top_expensive = df.nlargest(5, price_columns[-1])[['RegionName', 'State', price_columns[-1]]]
        y_pos = range(len(top_expensive))
        ax7.barh(y_pos, top_expensive[price_columns[-1]], color=self.colors[5])
        ax7.set_title('Top 5 Régions les Plus Chères', fontweight='bold')
        ax7.set_xlabel('Prix ($)')
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels([f"{row['RegionName']} ({row['State']})" 
                           for _, row in top_expensive.iterrows()])
        ax7.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Tableau de Bord - Analyse Zillow Home Value Index', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Fonction principale pour tester le module de visualisation
    """
    # Exemple d'utilisation
    print("🎨 Module de visualisation Zillow")
    print("Utilisez cette classe pour créer des visualisations des données Zillow")
    
    # Exemple de création d'un visualiseur
    viz = ZillowVisualizer()
    print("✅ Visualiseur initialisé avec succès")


if __name__ == "__main__":
    main()
