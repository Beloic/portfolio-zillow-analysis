"""
Module de modélisation pour les données Zillow Home Value Index
Auteur: Loic Bernard
Date: Décembre 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, List, Dict, Any
import joblib

warnings.filterwarnings('ignore')


class ZillowPredictor:
    """
    Classe pour créer et entraîner des modèles de prédiction des prix immobiliers
    """
    
    def __init__(self):
        """
        Initialise le prédicteur
        """
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.results = {}
        
    def prepare_features(self, 
                        df: pd.DataFrame, 
                        price_columns: List[str],
                        target_period: str = 'latest') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les caractéristiques pour l'entraînement
        
        Args:
            df (pd.DataFrame): Dataset avec métriques
            price_columns (List[str]): Colonnes de prix
            target_period (str): Période cible ('latest', 'future', ou date spécifique)
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        print("🔧 Préparation des caractéristiques...")
        
        # Copie du dataframe
        df_features = df.copy()
        
        # Encodage des variables catégorielles
        categorical_features = ['State', 'RegionType', 'Metro']
        for feature in categorical_features:
            if feature in df_features.columns:
                le = LabelEncoder()
                df_features[f'{feature}_encoded'] = le.fit_transform(df_features[feature].astype(str))
                self.encoders[feature] = le
        
        # Sélection des caractéristiques numériques
        numeric_features = [
            'prix_initial', 'croissance_totale', 'croissance_annuelle', 
            'volatilite', 'prix_max', 'prix_min', 'drawdown_max'
        ]
        
        # Ajout des caractéristiques encodées
        encoded_features = [f'{feat}_encoded' for feat in categorical_features if feat in df_features.columns]
        
        # Caractéristiques finales
        feature_columns = numeric_features + encoded_features
        
        # Filtrage des colonnes existantes
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        X = df_features[available_features].copy()
        
        # Définition de la cible
        if target_period == 'latest':
            y = df_features[price_columns[-1]]
        elif target_period == 'future':
            # Prédiction de la croissance future basée sur les tendances
            y = df_features['croissance_totale']
        else:
            y = df_features[target_period]
        
        # Suppression des valeurs manquantes
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Caractéristiques préparées: {X.shape[1]} features, {len(X)} échantillons")
        
        return X, y
    
    def train_models(self, 
                    X: pd.DataFrame, 
                    y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict[str, Any]:
        """
        Entraîne plusieurs modèles de régression
        
        Args:
            X (pd.DataFrame): Caractéristiques
            y (pd.Series): Variable cible
            test_size (float): Proportion pour le test
            random_state (int): Seed aléatoire
            
        Returns:
            Dict[str, Any]: Résultats des modèles
        """
        print("🤖 Entraînement des modèles...")
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Définition des modèles
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"  📊 Entraînement: {name}")
            
            # Entraînement
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Métriques
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Validation croisée
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Stockage des résultats
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            # Importance des caractéristiques (pour les modèles d'ensemble)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            print(f"    ✅ R² = {r2:.4f}, RMSE = {rmse:.2f}")
        
        self.models = {name: results[name]['model'] for name in results.keys()}
        self.results = results
        
        print("✅ Entraînement terminé")
        return results
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Affiche la comparaison des modèles
        
        Args:
            save_path (str): Chemin pour sauvegarder
        """
        if not self.results:
            print("❌ Aucun modèle entraîné. Utilisez train_models() d'abord.")
            return
        
        # Préparation des données
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2'] for name in model_names]
        rmse_scores = [self.results[name]['rmse'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        
        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Comparaison R²
        bars1 = axes[0, 0].bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Comparaison des Modèles - R² Score', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Comparaison RMSE
        bars2 = axes[0, 1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Comparaison des Modèles - RMSE', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, rmse_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + score*0.01,
                          f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Validation croisée
        bars3 = axes[1, 0].bar(model_names, cv_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Validation Croisée - R² Score', fontweight='bold')
        axes[1, 0].set_ylabel('R² Score (CV)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars3, cv_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prédictions vs Réalité (meilleur modèle)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_result = self.results[best_model_name]
        
        axes[1, 1].scatter(best_result['y_test'], best_result['y_pred'], alpha=0.6, color='purple')
        axes[1, 1].plot([best_result['y_test'].min(), best_result['y_test'].max()], 
                       [best_result['y_test'].min(), best_result['y_test'].max()], 
                       'r--', alpha=0.8, label='Prédiction parfaite')
        axes[1, 1].set_title(f'Prédictions vs Réalité - {best_model_name}', fontweight='bold')
        axes[1, 1].set_xlabel('Valeurs Réelles')
        axes[1, 1].set_ylabel('Prédictions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, 
                               model_name: str = None,
                               top_n: int = 10,
                               save_path: str = None) -> None:
        """
        Affiche l'importance des caractéristiques
        
        Args:
            model_name (str): Nom du modèle (si None, utilise le meilleur)
            top_n (int): Nombre de caractéristiques à afficher
            save_path (str): Chemin pour sauvegarder
        """
        if not self.feature_importance:
            print("❌ Aucune information sur l'importance des caractéristiques disponible.")
            return
        
        if model_name is None:
            # Utilise le modèle avec le meilleur R²
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
            model_name = best_model_name
        
        if model_name not in self.feature_importance:
            print(f"❌ Modèle '{model_name}' non trouvé.")
            return
        
        # Récupération des importances
        importances = self.feature_importance[model_name]
        
        # Tri par importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance_values = zip(*sorted_features)
        
        # Création du graphique
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importance_values, color='steelblue', alpha=0.7)
        
        plt.title(f'Importance des Caractéristiques - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Caractéristiques')
        plt.yticks(range(len(features)), features)
        
        # Ajouter les valeurs
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Retourne le meilleur modèle basé sur le R² score
        
        Returns:
            Tuple[str, Any]: Nom et modèle du meilleur modèle
        """
        if not self.results:
            raise ValueError("Aucun modèle entraîné. Utilisez train_models() d'abord.")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Sauvegarde un modèle entraîné
        
        Args:
            model_name (str): Nom du modèle
            filepath (str): Chemin de sauvegarde
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé.")
        
        joblib.dump({
            'model': self.models[model_name],
            'scaler': self.scalers.get('main'),
            'encoders': self.encoders,
            'results': self.results[model_name]
        }, filepath)
        
        print(f"✅ Modèle '{model_name}' sauvegardé: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath (str): Chemin du modèle
        """
        data = joblib.load(filepath)
        
        self.models = {'loaded_model': data['model']}
        self.scalers = {'main': data['scaler']}
        self.encoders = data['encoders']
        self.results = {'loaded_model': data['results']}
        
        print(f"✅ Modèle chargé: {filepath}")


def main():
    """
    Fonction principale pour tester le module de modélisation
    """
    print("🤖 Module de modélisation Zillow")
    print("Utilisez cette classe pour créer et entraîner des modèles de prédiction")
    
    # Exemple d'utilisation
    predictor = ZillowPredictor()
    print("✅ Prédicteur initialisé avec succès")


if __name__ == "__main__":
    main()
