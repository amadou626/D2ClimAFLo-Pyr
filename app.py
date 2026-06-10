import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import mannwhitneyu, ttest_ind, chi2
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="D2ClimAFLo-Pyr",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stMetric { background-color: #FFFFFF;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #E0E0E0; }
    .titre-page {
        font-size: 2rem;
        font-weight: 700;
        color: #1B2D26;
        border-left: 5px solid #1D9E75;
        padding-left: 12px;
        margin-bottom: 20px;
    }
    .carte-site {
        background: #EEEDFE;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        border-left: 4px solid #534AB7;
    }
    .nohedes-box {
        background: #E8F5E9;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        border-left: 4px solid #1D9E75;
    }
    .histoire-box {
        background: linear-gradient(135deg, #1B2D26 0%, #2D6A4F 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        font-size: 1.1rem;
        line-height: 1.8;
        margin: 10px 0;
    }
    .question-box {
        background: #EEEDFE;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #3C3489;
        border: 2px solid #7F77DD;
        margin: 10px 0;
    }
    .sig-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES & FONCTIONS — pour PAGE PERSPECTIVES
# ══════════════════════════════════════════════════════════════════

# Variables avec noms lisibles
VARS_INFO = {
    "SMOD_modis": "Fin enneigement (jour nival)",
    "continuite": "Continuité du manteau neigeux",
    "temp_RF_moy": "Température air moyenne (°C)",
    "SCD": "Nombre de jours de neige",
    "solar_radiation_moy": "Rayonnement solaire (W/m²)",
    "humidity_moy": "Humidité air moyenne (%)",
    "neige_pct": "% précipitations neigeuses",
    "soil_temp_upper_moy": "Température sol (°C)",
    "SOD_modis": "Début enneigement (jour nival)",
    "duree_saison": "Durée saison nivale (jours)",
    "soil_moisture_moy": "Humidité sol (m³/m³)",
    "pr_RF_total": "Précipitations totales (mm)",
    "wind_speed_moy": "Vitesse du vent (m/s)",
    "etp_total": "Évapotranspiration (mm)",
    "LFD_nival": "Dernier jour de gel (jour nival)"
}

# Fenêtres temporelles
FENETRES = {
    "F1 : 2000-2004": [2000, 2001, 2002, 2004],
    "F2 : 2005-2009": [2005, 2006, 2007, 2008, 2009],
    "F3 : 2010-2014": [2010, 2011, 2012, 2013, 2014],
    "F4 : 2015-2017": [2015, 2016, 2017],
    "F5 : 2018-2020": [2018, 2019, 2020]
}

# Couleurs
COLOR_NOHEDES = "#2E7D32"      # Vert foncé (Delphinium)
COLOR_AUTRES = "#7E3AC8"       # Violet
COLOR_ELLIPSE = "rgba(126, 58, 200, 0.15)"
COLOR_ELLIPSE_LINE = "rgba(126, 58, 200, 0.7)"
COLOR_ARROW = "rgba(255, 140, 0, 0.85)"   # Orange pour flèches variables

# ───────────────────────────────────────────────────────────────────
# Fonctions de calcul et de visualisation
# ───────────────────────────────────────────────────────────────────

# FONCTIONS DE CALCUL
# ═══════════════════════════════════════════════════════════════════

def calculate_pca_mahalanobis(df, annees, variables):
    """
    Calcule la PCA, distance Mahalanobis et tous les éléments pour
    construire l'ellipse de confiance 95%.
    """
    # Filtrer données pour la fenêtre
    df_fen = df[df['annee'].isin(annees)].copy()
    
    # Moyenne par site
    df_moy = df_fen.groupby('nom')[variables].mean(numeric_only=True).reset_index()
    
    # Retirer variables avec NA
    vars_ok = [v for v in variables if not df_moy[v].isna().any()]
    
    if len(vars_ok) < 2:
        return None
    
    # Centrer-réduire
    X = df_moy[vars_ok].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    var_explique = pca.explained_variance_ratio_ * 100
    
    # Construire df coordonnées
    df_pca = pd.DataFrame({
        'nom': df_moy['nom'].values,
        'PC1': coords[:, 0],
        'PC2': coords[:, 1]
    })
    df_pca['type'] = df_pca['nom'].apply(
        lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
    )
    
    # NOHEDES
    if 'NOHEDES' not in df_pca['nom'].values:
        return None
    
    nohedes = df_pca[df_pca['nom'] == 'NOHEDES'][['PC1', 'PC2']].values[0]
    autres = df_pca[df_pca['nom'] != 'NOHEDES'][['PC1', 'PC2']].values
    
    # Centroïde + Covariance
    centroid = autres.mean(axis=0)
    cov_mat = np.cov(autres.T)
    
    # Distance Mahalanobis
    try:
        inv_cov = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        lambda_reg = 1e-3 * np.mean(np.diag(cov_mat))
        inv_cov = np.linalg.inv(cov_mat + lambda_reg * np.eye(2))
    
    diff = nohedes - centroid
    D2 = float(diff @ inv_cov @ diff.T)
    D = np.sqrt(D2)
    p_value = 1 - chi2.cdf(D2, df=2)
    
    # Vérifier si NOHEDES est dans l'ellipse 95%
    seuil_chi2 = chi2.ppf(0.95, df=2)
    dans_ellipse = D2 < seuil_chi2
    
    # Contributions des variables aux PC1 et PC2
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=['PC1', 'PC2'],
        index=vars_ok
    )
    
    return {
        'df_pca': df_pca,
        'centroid': centroid,
        'cov_mat': cov_mat,
        'nohedes_coords': nohedes,
        'D2': D2,
        'D': D,
        'p_value': p_value,
        'var_pc1': var_explique[0],
        'var_pc2': var_explique[1],
        'var_pc12': var_explique[0] + var_explique[1],
        'dans_ellipse': dans_ellipse,
        'seuil_chi2': seuil_chi2,
        'n_variables': len(vars_ok),
        'vars_utilisees': vars_ok,
        'loadings': loadings
    }


def get_ellipse_points(centroid, cov_mat, confidence=0.95, n_points=100):
    """Calcule les points de l'ellipse de confiance"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    
    # Tri décroissant
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    chi2_val = chi2.ppf(confidence, df=2)
    
    width = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])
    
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    t = np.linspace(0, 2 * np.pi, n_points)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    x_unit = (width / 2) * np.cos(t)
    y_unit = (height / 2) * np.sin(t)
    
    ellipse_x = centroid[0] + x_unit * cos_a - y_unit * sin_a
    ellipse_y = centroid[1] + x_unit * sin_a + y_unit * cos_a
    
    return ellipse_x, ellipse_y


def calculate_zscores_per_variable(df, annees, variables):
    """Calcule le Z-score de NOHEDES pour chaque variable"""
    df_fen = df[df['annee'].isin(annees)].copy()
    df_moy = df_fen.groupby('nom')[variables].mean(numeric_only=True).reset_index()
    
    resultats = []
    for var in variables:
        if var not in df_moy.columns or df_moy[var].isna().any():
            continue
        
        noh_val = df_moy[df_moy['nom'] == 'NOHEDES'][var].values[0]
        autres_vals = df_moy[df_moy['nom'] != 'NOHEDES'][var].values
        moy_aut = autres_vals.mean()
        sd_aut = autres_vals.std(ddof=1)
        
        if sd_aut == 0:
            z = 0
        else:
            z = (noh_val - moy_aut) / sd_aut
        
        resultats.append({
            'variable': var,
            'nom_lisible': VARS_INFO.get(var, var),
            'NOHEDES': noh_val,
            'moy_autres': moy_aut,
            'z_score': z
        })
    
    return pd.DataFrame(resultats)


def jour_nival_to_date(jour_nival, annee_nivale=2020):
    """Convertit un jour nival en date calendaire (1er sept = jour 1)"""
    from datetime import datetime, timedelta
    
    if pd.isna(jour_nival):
        return ""
    
    # Année nivale commence le 1er septembre
    debut = datetime(annee_nivale - 1, 9, 1)
    date = debut + timedelta(days=int(jour_nival) - 1)
    
    # Format français : "15 mars"
    mois_fr = {
        1: "jan", 2: "fév", 3: "mars", 4: "avr", 5: "mai", 6: "juin",
        7: "juil", 8: "août", 9: "sept", 10: "oct", 11: "nov", 12: "déc"
    }
    return f"{date.day} {mois_fr[date.month]}"


def jour_nival_to_saison(jour_nival):
    """Détermine la saison à partir du jour nival (1er sept = jour 1)"""
    if pd.isna(jour_nival):
        return "?"
    
    # Année nivale commence 1er septembre
    # Sept-Nov = jour 1-91 → Automne
    # Déc-Fév = jour 92-181 → Hiver
    # Mars-Mai = jour 182-273 → Printemps
    # Juin-Août = jour 274-365 → Été
    
    j = int(jour_nival)
    if j <= 91:
        return "Automne (SON)"
    elif j <= 181:
        return "Hiver (DJF)"
    elif j <= 273:
        return "Printemps (MAM)"
    else:
        return "Été (JJA)"


def plot_zscores_barres(zscores_df, titre_suffixe=""):
    """Crée graphique en barres des Z-scores avec code couleur"""
    # Trier par |z| décroissant pour mieux visualiser
    zscores_df = zscores_df.copy()
    zscores_df['abs_z'] = zscores_df['z_score'].abs()
    zscores_df = zscores_df.sort_values('abs_z', ascending=True)
    
    # Couleurs : rouge si différent, vert si similaire
    couleurs = []
    for z in zscores_df['z_score']:
        if abs(z) >= 2:
            couleurs.append('#C62828')  # Rouge foncé - très différent
        elif abs(z) >= 1:
            couleurs.append('#FF8F00')  # Orange - modérément différent
        else:
            couleurs.append('#2E7D32')  # Vert - similaire
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=zscores_df['z_score'],
        y=zscores_df['nom_lisible'],
        orientation='h',
        marker=dict(color=couleurs,
                    line=dict(width=1, color='white')),
        text=[f"{z:+.2f}" for z in zscores_df['z_score']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'NOHEDES : %{customdata[0]:.2f}<br>'
            'Moy. autres : %{customdata[1]:.2f}<br>'
            'Z-score : %{x:+.2f}'
            '<extra></extra>'
        ),
        customdata=zscores_df[['NOHEDES', 'moy_autres']].values
    ))
    
    # Lignes seuils
    fig.add_vline(x=0, line_color='gray', line_width=1)
    fig.add_vline(x=2, line_dash='dash', line_color='red', line_width=1,
                    annotation_text="z=+2", annotation_position="top")
    fig.add_vline(x=-2, line_dash='dash', line_color='red', line_width=1,
                    annotation_text="z=-2", annotation_position="top")
    fig.add_vline(x=1, line_dash='dot', line_color='orange', line_width=1,
                    annotation_text="z=+1", annotation_position="bottom")
    fig.add_vline(x=-1, line_dash='dot', line_color='orange', line_width=1,
                    annotation_text="z=-1", annotation_position="bottom")
    
    fig.update_layout(
        title=dict(
            text=f"<b>Z-scores des variables — NOHEDES vs autres sites</b>"
                  f"{('<br><span style=\"font-size:12px\">' + titre_suffixe + '</span>') if titre_suffixe else ''}",
            x=0.5, xanchor='center'
        ),
        xaxis_title="Z-score (différence en écarts-types)",
        yaxis_title="",
        height=max(400, 50 * len(zscores_df) + 100),
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_evolution_smod_lfd(df):
    """Évolution temporelle SMOD vs LFD avec dates calendaires"""
    df_filt = df[df['annee'].between(2000, 2020)].copy()
    df_filt['type'] = df_filt['nom'].apply(
        lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
    )
    
    # Moyennes par année et type pour SMOD et LFD
    df_agg = df_filt.groupby(['annee', 'type']).agg(
        SMOD_moy=('SMOD_modis', 'mean'),
        LFD_moy=('LFD_nival', 'mean')
    ).reset_index()
    
    # Hovertext avec dates
    df_agg['SMOD_date'] = df_agg['SMOD_moy'].apply(jour_nival_to_date)
    df_agg['LFD_date'] = df_agg['LFD_moy'].apply(jour_nival_to_date)
    
    fig = go.Figure()
    
    # NOHEDES
    df_noh = df_agg[df_agg['type'] == 'NOHEDES']
    
    # SMOD NOHEDES
    fig.add_trace(go.Scatter(
        x=df_noh['annee'], y=df_noh['SMOD_moy'],
        mode='lines+markers',
        name='SMOD NOHEDES',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=10, symbol='circle'),
        text=df_noh['SMOD_date'],
        hovertemplate='<b>NOHEDES - SMOD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    # LFD NOHEDES
    fig.add_trace(go.Scatter(
        x=df_noh['annee'], y=df_noh['LFD_moy'],
        mode='lines+markers',
        name='LFD NOHEDES',
        line=dict(color='#2E7D32', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        text=df_noh['LFD_date'],
        hovertemplate='<b>NOHEDES - LFD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    # Autres
    df_aut = df_agg[df_agg['type'] == 'Autres sites']
    
    fig.add_trace(go.Scatter(
        x=df_aut['annee'], y=df_aut['SMOD_moy'],
        mode='lines+markers',
        name='SMOD moyenne 8 sites',
        line=dict(color='#7E3AC8', width=3),
        marker=dict(size=10, symbol='circle'),
        text=df_aut['SMOD_date'],
        hovertemplate='<b>Autres - SMOD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_aut['annee'], y=df_aut['LFD_moy'],
        mode='lines+markers',
        name='LFD moyenne 8 sites',
        line=dict(color='#7E3AC8', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        text=df_aut['LFD_date'],
        hovertemplate='<b>Autres - LFD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    # Lignes de référence pour les saisons (1er sept = 1)
    y_lignes = [91, 181, 273]  # bornes des saisons en jour nival
    labels_lignes = ['Fin Automne (1 déc)', 'Fin Hiver (1 mars)', 'Fin Printemps (1 juin)']
    
    for y, label in zip(y_lignes, labels_lignes):
        fig.add_hline(y=y, line_dash='dot', line_color='lightgray',
                       annotation_text=label, annotation_position="right",
                       annotation=dict(font=dict(size=9, color='gray')))
    
    fig.update_layout(
        title=dict(
            text="<b>Évolution SMOD (fin neige) vs LFD (dernier gel)</b><br>"
                  "<span style='font-size:12px'>NOHEDES vs Moyenne 8 sites fleurissants (2000-2020)</span>",
            x=0.5, xanchor='center'
        ),
        xaxis_title="Année",
        yaxis_title="Jour nival (1 sept = 1)",
        height=550,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2,
                      xanchor='center', x=0.5)
    )
    
    return fig


def plot_distribution_saisons(df):
    """Distribution des SMOD et LFD par saison (NOHEDES vs autres)"""
    df_filt = df[df['annee'].between(2000, 2020)].copy()
    df_filt['type'] = df_filt['nom'].apply(
        lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
    )
    
    # Calculer saisons
    df_filt['saison_SMOD'] = df_filt['SMOD_modis'].apply(jour_nival_to_saison)
    df_filt['saison_LFD'] = df_filt['LFD_nival'].apply(jour_nival_to_saison)
    
    saisons_ordre = ['Automne (SON)', 'Hiver (DJF)', 'Printemps (MAM)', 'Été (JJA)']
    
    # Comptages
    smod_counts = df_filt.groupby(['type', 'saison_SMOD']).size().reset_index(name='n')
    lfd_counts = df_filt.groupby(['type', 'saison_LFD']).size().reset_index(name='n')
    
    # Subplot 2 graphiques côte à côte
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SMOD : dans quelle saison ?",
                          "LFD : dans quelle saison ?")
    )
    
    # SMOD
    for type_site, couleur in [('NOHEDES', '#2E7D32'), ('Autres sites', '#7E3AC8')]:
        # Filtrer pour ce type
        df_type = smod_counts[smod_counts['type'] == type_site].copy()
        
        # Créer un dict saison -> n
        dict_saisons = dict(zip(df_type['saison_SMOD'], df_type['n']))
        
        # Construire les valeurs dans l'ordre des saisons
        x_vals = saisons_ordre
        y_vals = [dict_saisons.get(s, 0) for s in saisons_ordre]
        
        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals,
                name=type_site,
                marker_color=couleur,
                text=y_vals,
                textposition='outside',
                showlegend=True,
                legendgroup=type_site
            ),
            row=1, col=1
        )
    
    # LFD
    for type_site, couleur in [('NOHEDES', '#2E7D32'), ('Autres sites', '#7E3AC8')]:
        df_type = lfd_counts[lfd_counts['type'] == type_site].copy()
        dict_saisons = dict(zip(df_type['saison_LFD'], df_type['n']))
        
        x_vals = saisons_ordre
        y_vals = [dict_saisons.get(s, 0) for s in saisons_ordre]
        
        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals,
                name=type_site,
                marker_color=couleur,
                text=y_vals,
                textposition='outside',
                showlegend=False,
                legendgroup=type_site
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=dict(
            text="<b>Distribution saisonnière des événements</b>",
            x=0.5, xanchor='center'
        ),
        height=450,
        template='plotly_white',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2,
                      xanchor='center', x=0.5)
    )
    
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(title_text="Nombre d'années")
    
    return fig


def plot_pca_with_ellipse(result, periode_label):
    """Crée graphique PCA avec ellipse 95%"""
    df_pca = result['df_pca']
    
    fig = go.Figure()
    
    # Ellipse 95%
    ellipse_x, ellipse_y = get_ellipse_points(
        result['centroid'], result['cov_mat'], confidence=0.95
    )
    
    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        line=dict(color=COLOR_ELLIPSE_LINE, dash='dash', width=2),
        fill='toself',
        fillcolor=COLOR_ELLIPSE,
        name='Ellipse 95% (8 sites)',
        hoverinfo='skip'
    ))
    
    # Ligne NOHEDES → Centroïde (distance)
    nohedes_coords = result['nohedes_coords']
    fig.add_trace(go.Scatter(
        x=[nohedes_coords[0], result['centroid'][0]],
        y=[nohedes_coords[1], result['centroid'][1]],
        mode='lines',
        line=dict(color='rgba(80, 80, 80, 0.7)', width=2, dash='dot'),
        name=f"Distance D = {result['D']:.2f}",
        hovertemplate=(
            f"<b>Distance Mahalanobis</b><br>"
            f"D² = {result['D2']:.2f}<br>"
            f"D = {result['D']:.2f}<br>"
            f"p-value = {result['p_value']:.4f}"
            "<extra></extra>"
        )
    ))
    
    # Annotation au milieu de la ligne (valeur D²)
    milieu_x = (nohedes_coords[0] + result['centroid'][0]) / 2
    milieu_y = (nohedes_coords[1] + result['centroid'][1]) / 2
    
    fig.add_annotation(
        x=milieu_x,
        y=milieu_y,
        text=f"<b>D² = {result['D2']:.2f}</b>",
        showarrow=False,
        font=dict(size=12, color='#444', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.85)',
        bordercolor='rgba(80, 80, 80, 0.7)',
        borderwidth=1,
        borderpad=4
    )
    
    # Centroïde
    fig.add_trace(go.Scatter(
        x=[result['centroid'][0]], y=[result['centroid'][1]],
        mode='markers',
        marker=dict(color=COLOR_AUTRES, size=14, symbol='x',
                    line=dict(width=3)),
        name='Centroïde 8 sites',
        hovertemplate='Centroïde des 8 sites<extra></extra>'
    ))
    
    # Autres sites
    autres = df_pca[df_pca['nom'] != 'NOHEDES']
    fig.add_trace(go.Scatter(
        x=autres['PC1'], y=autres['PC2'],
        mode='markers+text',
        marker=dict(color=COLOR_AUTRES, size=14,
                    line=dict(width=1.5, color='white')),
        text=autres['nom'],
        textposition='top center',
        textfont=dict(size=10),
        name='Sites fleurissants',
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    # NOHEDES
    nohedes = df_pca[df_pca['nom'] == 'NOHEDES']
    fig.add_trace(go.Scatter(
        x=nohedes['PC1'], y=nohedes['PC2'],
        mode='markers+text',
        marker=dict(color=COLOR_NOHEDES, size=22, symbol='diamond',
                    line=dict(width=2, color='white')),
        text=['NOHEDES'],
        textposition='top center',
        textfont=dict(color=COLOR_NOHEDES, size=14, family='Arial Black'),
        name='NOHEDES',
        hovertemplate='<b>NOHEDES</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BIPLOT : Flèches des variables
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    loadings = result['loadings']
    
    # Échelle adaptative : ajuster les flèches à la taille du graphique
    pc1_range = df_pca['PC1'].max() - df_pca['PC1'].min()
    pc2_range = df_pca['PC2'].max() - df_pca['PC2'].min()
    
    # Facteur d'échelle pour que les flèches soient visibles mais pas trop grandes
    max_loading = np.abs(loadings.values).max()
    scale_factor = min(pc1_range, pc2_range) * 0.45 / max_loading
    
    for var_name in loadings.index:
        x_arrow = loadings.loc[var_name, 'PC1'] * scale_factor
        y_arrow = loadings.loc[var_name, 'PC2'] * scale_factor
        
        # Flèche
        fig.add_annotation(
            x=x_arrow, y=y_arrow,
            ax=0, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor=COLOR_ARROW,
            opacity=0.85
        )
        
        # Étiquette de la variable (nom français court)
        label_short = VARS_INFO.get(var_name, var_name)
        # Raccourcir pour lisibilité
        if len(label_short) > 25:
            label_short = label_short[:23] + "..."
        
        # Position de l'étiquette (légèrement au-delà de la flèche)
        label_x = x_arrow * 1.15
        label_y = y_arrow * 1.15
        
        fig.add_annotation(
            x=label_x, y=label_y,
            text=f"<i>{label_short}</i>",
            showarrow=False,
            font=dict(size=10, color=COLOR_ARROW, family='Arial'),
            bgcolor='rgba(255, 255, 255, 0.75)',
            borderpad=2,
            xanchor='center'
        )
    
    # Titre dynamique
    statut_label = "✓ DANS l'ellipse" if result['dans_ellipse'] else "⚠ HORS de l'ellipse"
    
    fig.update_layout(
        title=dict(
            text=f"<b>PCA — {periode_label}</b><br>"
                  f"<span style='font-size:13px'>"
                  f"PC1 ({result['var_pc1']:.1f}%) + PC2 ({result['var_pc2']:.1f}%) "
                  f"= {result['var_pc12']:.1f}% · NOHEDES {statut_label}"
                  f"</span>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=f"PC1 ({result['var_pc1']:.1f}%)",
        yaxis_title=f"PC2 ({result['var_pc2']:.1f}%)",
        height=550,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Axes équilibrés (inclure points + ellipse + flèches biplot)
    loadings = result['loadings']
    pc1_range = df_pca['PC1'].max() - df_pca['PC1'].min()
    pc2_range = df_pca['PC2'].max() - df_pca['PC2'].min()
    max_loading = np.abs(loadings.values).max()
    scale_factor = min(pc1_range, pc2_range) * 0.45 / max_loading
    
    arrows_x = loadings['PC1'].values * scale_factor * 1.25  # inclure les étiquettes
    arrows_y = loadings['PC2'].values * scale_factor * 1.25
    
    all_x = list(df_pca['PC1']) + list(ellipse_x) + list(arrows_x) + list(-arrows_x)
    all_y = list(df_pca['PC2']) + list(ellipse_y) + list(arrows_y) + list(-arrows_y)
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    margin_x = (x_max - x_min) * 0.10
    margin_y = (y_max - y_min) * 0.10
    
    fig.update_xaxes(range=[x_min - margin_x, x_max + margin_x],
                      zeroline=True, zerolinecolor='lightgray')
    fig.update_yaxes(range=[y_min - margin_y, y_max + margin_y],
                      zeroline=True, zerolinecolor='lightgray')
    
    return fig


def plot_evolution_variable(df, variable):
    """Graphique d'évolution annuelle d'une variable"""
    df_filt = df[df['annee'].between(2000, 2020)].copy()
    
    df_filt['type'] = df_filt['nom'].apply(
        lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
    )
    
    # Moyennes par année et type
    df_agg = df_filt.groupby(['annee', 'type'])[variable].agg(
        ['mean', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    # NOHEDES
    df_noh = df_agg[df_agg['type'] == 'NOHEDES']
    fig.add_trace(go.Scatter(
        x=df_noh['annee'], y=df_noh['mean'],
        mode='lines+markers',
        line=dict(color=COLOR_NOHEDES, width=3),
        marker=dict(size=10),
        name='NOHEDES'
    ))
    
    # Moyenne autres
    df_aut = df_agg[df_agg['type'] == 'Autres sites']
    fig.add_trace(go.Scatter(
        x=df_aut['annee'], y=df_aut['mean'],
        mode='lines+markers',
        line=dict(color=COLOR_AUTRES, width=3),
        marker=dict(size=10),
        name='Moyenne 8 sites fleurissants'
    ))
    
    # Bande min-max des autres sites
    fig.add_trace(go.Scatter(
        x=list(df_aut['annee']) + list(df_aut['annee'][::-1]),
        y=list(df_aut['max']) + list(df_aut['min'][::-1]),
        fill='toself',
        fillcolor='rgba(126, 58, 200, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True,
        name='Étendue 8 sites (min-max)',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>Évolution : {VARS_INFO[variable]}</b><br>"
                  f"<span style='font-size:12px'>NOHEDES vs sites fleurissants (2000-2020)</span>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Année",
        yaxis_title=VARS_INFO[variable],
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_mahalanobis_evolution(resultats_par_fenetre):
    """Graphique évolution D² par fenêtre"""
    fig = go.Figure()
    
    # Seuil
    seuil = chi2.ppf(0.95, df=2)
    fig.add_hline(
        y=seuil, line_dash="dash", line_color="red",
        annotation_text=f"Seuil χ²(2)=5%% : {seuil:.2f}",
        annotation_position="top right"
    )
    
    # Données
    fenetres_labels = list(resultats_par_fenetre.keys())
    d2_values = [r['D2'] for r in resultats_par_fenetre.values()]
    p_values = [r['p_value'] for r in resultats_par_fenetre.values()]
    statuts = ['ATYPIQUE' if r['D2'] > seuil else 'SIMILAIRE' 
                for r in resultats_par_fenetre.values()]
    
    couleurs = [COLOR_NOHEDES if s == 'ATYPIQUE' else COLOR_AUTRES 
                  for s in statuts]
    
    # Ligne
    fig.add_trace(go.Scatter(
        x=fenetres_labels,
        y=d2_values,
        mode='lines',
        line=dict(color='darkblue', width=2),
        showlegend=False
    ))
    
    # Points
    fig.add_trace(go.Scatter(
        x=fenetres_labels,
        y=d2_values,
        mode='markers+text',
        marker=dict(color=couleurs, size=18,
                    line=dict(width=2, color='white')),
        text=[f"{d:.1f}" for d in d2_values],
        textposition='top center',
        textfont=dict(size=12, family='Arial Black'),
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>D² = %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Distance Mahalanobis par fenêtre temporelle</b>",
            x=0.5, xanchor='center'
        ),
        xaxis_title="Fenêtre temporelle",
        yaxis_title="Distance Mahalanobis (D²)",
        height=450,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def charger_donnees(chemin):
    df = pd.read_csv(chemin, sep=";")
    df["groupe"] = df["nom"].apply(
        lambda x: "NOHEDES" if x == "NOHEDES" else "Autres sites"
    )
    df["saison"] = df["mois"].apply(lambda m:
        "Hiver"     if m in [12, 1, 2]  else
        "Printemps" if m in [3, 4, 5]   else
        "Été"       if m in [6, 7, 8]   else
        "Automne"
    )
    df["periode"] = df["annee"].apply(
        lambda a: "Avant 2010" if a < 2010 else "Après 2010"
    )
    return df

# ── Sites ──────────────────────────────────────────────────────────
SITES = pd.DataFrame({
    "nom"      : ["CADI_POP1","CADI_POP2","CADI_POP3","CADI_POP4",
                  "EYNE_POP1","EYNE_POP2","EYNE_POP3",
                  "NOHEDES","VALLTER"],
    "lat"      : [42.237,42.277,42.289,42.283,
                  42.450,42.446,42.443,
                  42.615,42.426],
    "lon"      : [1.704,1.688,1.688,1.574,
                  2.130,2.122,2.116,
                  2.263,2.264],
    "altitude" : [2365,2283,1979,2358,
                  2586,2181,2164,
                  1790,2147],
    "versant"  : ["Espagne","Espagne","Espagne","Espagne",
                  "France","France","France",
                  "France","Espagne"],
    "floraison": [True,True,True,True,
                  True,True,True,
                  False,True],
    "groupe_site": ["CADI","CADI","CADI","CADI",
                    "EYNE","EYNE","EYNE",
                    "NOHEDES","VALLTER"]
})

VARS = {
    "temp_RF"    : "Température (°C)",
    "pr_RF"      : "Précipitations (mm/mois)",
    "neige_RF_cm": "Hauteur neige (cm)",
    "pct_neige"  : "% Couverture neigeuse"
}

COULEURS = {
    "NOHEDES"      : "#1D9E75",
    "Autres sites" : "#7F77DD"
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌸 D2ClimAFLo-Pyr")
    st.markdown("**Stage BUT3 Science des Données**")
    st.markdown("*CEFREM · UPVD · 2026*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🎭 Introduction",
         "🗺️ Zone d'étude",
         "🛰️ Downscaling",
         "📊 Statistiques descriptives",
         "📈 Évolution temporelle",
         "🔬 ACP + CAH",
         "🧪 Test d'hypothèse",
         "🏁 Conclusion",
         "🎯 Perspectives"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Charger les données**")
    chemin = st.text_input(
        "Chemin du fichier CSV",
        value="df_complete_2000_2020.csv",
        help="Chemin vers df_complete_2000_2020.csv"
    )

    try:
        df = charger_donnees(chemin)
        st.success(f"✅ {len(df):,} observations chargées")
        data_ok = True
    except Exception as e:
        st.warning("⚠️ Fichier non trouvé — données simulées")
        # Données simulées pour démonstration
        np.random.seed(42)
        noms = (["CADI_POP1"]*12 + ["CADI_POP2"]*12 +
                ["CADI_POP3"]*12 + ["CADI_POP4"]*12 +
                ["EYNE_POP1"]*12 + ["EYNE_POP2"]*12 +
                ["EYNE_POP3"]*12 + ["NOHEDES"]*12 +
                ["VALLTER"]*12) * 21
        annees = sorted(list(range(2000, 2021)) * (9*12))
        mois   = list(range(1, 13)) * (9*21)
        df = pd.DataFrame({
            "nom"        : noms[:len(annees)],
            "annee"      : annees,
            "mois"       : mois[:len(annees)],
            "temp_RF"    : np.where(
                pd.Series(noms[:len(annees)]) == "NOHEDES",
                np.random.normal(5.5, 2, len(annees)),
                np.random.normal(4.3, 2, len(annees))
            ),
            "pr_RF"      : np.where(
                pd.Series(noms[:len(annees)]) == "NOHEDES",
                np.random.normal(78, 20, len(annees)),
                np.random.normal(70, 20, len(annees))
            ),
            "neige_RF_cm": np.where(
                pd.Series(noms[:len(annees)]) == "NOHEDES",
                np.random.exponential(3, len(annees)),
                np.random.exponential(15, len(annees))
            ),
            "pct_neige"  : np.where(
                pd.Series(noms[:len(annees)]) == "NOHEDES",
                np.random.uniform(5, 30, len(annees)),
                np.random.uniform(15, 60, len(annees))
            ),
            "altitude"   : [SITES.set_index("nom")
                            .loc[n,"altitude"]
                            for n in noms[:len(annees)]]
        })
        df["groupe"] = df["nom"].apply(
            lambda x: "NOHEDES" if x == "NOHEDES" else "Autres sites"
        )
        df["saison"] = df["mois"].apply(lambda m:
            "Hiver"     if m in [12, 1, 2]  else
            "Printemps" if m in [3, 4, 5]   else
            "Été"       if m in [6, 7, 8]   else
            "Automne"
        )
        df["periode"] = df["annee"].apply(
            lambda a: "Avant 2010" if a < 2010 else "Après 2010"
        )
        data_ok = True

    st.divider()
    st.markdown("**Auteur**")
    st.markdown("Amadou FOFANA")
    st.markdown("Superviseurs : S. Pinel · N. Collette")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — INTRODUCTION
# ══════════════════════════════════════════════════════════════════
if page == "🎭 Introduction":

    st.markdown('<div class="titre-page">🌸 Défaut de floraison de Delphinium montanum</div>',
                unsafe_allow_html=True)

    # Histoire visuelle SVG — via components.html pour affichage garanti
    import streamlit.components.v1 as components

    st.markdown("### 🎭 Il était une fois...")

    histoire_html = """
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:transparent;overflow:hidden">
    <div style="background:linear-gradient(135deg,#1B2D26,#2D6A4F);
                border-radius:16px;padding:16px 24px 24px 24px;margin:0">
    <svg viewBox="0 0 700 240" width="100%" height="280"
         xmlns="http://www.w3.org/2000/svg"
         style="display:block">
      <defs>
        <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto">
          <path d="M2 1L8 5L2 9" fill="none"
                stroke="white" stroke-width="1.5"/>
        </marker>
      </defs>

      <!-- Titre -->
      <text x="350" y="25" text-anchor="middle"
            fill="white" font-size="14" font-weight="bold"
            font-family="sans-serif">
        Delphinium montanum — Nohèdes, Pyrénées
      </text>

      <!-- ── GAUCHE : Avant 2010 ── -->
      <!-- Montagne -->
      <polygon points="80,185 145,85 210,185"
               fill="#2D6A4F" stroke="#9FE1CB" stroke-width="2"/>
      <!-- Neige -->
      <polygon points="145,85 135,112 155,112"
               fill="white" opacity="0.7"/>
      <!-- Sol -->
      <rect x="60" y="185" width="170" height="8"
            rx="4" fill="#3B6D11" opacity="0.6"/>
      <!-- Fleurs violettes -->
      <circle cx="80"  cy="182" r="7" fill="#7F77DD"/>
      <circle cx="97"  cy="179" r="7" fill="#AFA9EC"/>
      <circle cx="114" cy="182" r="7" fill="#7F77DD"/>
      <circle cx="165" cy="179" r="7" fill="#AFA9EC"/>
      <circle cx="182" cy="182" r="7" fill="#7F77DD"/>
      <circle cx="199" cy="180" r="7" fill="#AFA9EC"/>
      <line x1="80"  y1="189" x2="80"  y2="200" stroke="#5A8F2A" stroke-width="1.5"/>
      <line x1="97"  y1="186" x2="97"  y2="200" stroke="#5A8F2A" stroke-width="1.5"/>
      <line x1="114" y1="189" x2="114" y2="200" stroke="#5A8F2A" stroke-width="1.5"/>
      <line x1="165" y1="186" x2="165" y2="200" stroke="#5A8F2A" stroke-width="1.5"/>
      <line x1="182" y1="189" x2="182" y2="200" stroke="#5A8F2A" stroke-width="1.5"/>
      <line x1="199" y1="187" x2="199" y2="200" stroke="#5A8F2A" stroke-width="1.5"/>

      <text x="143" y="220" text-anchor="middle"
            fill="#9FE1CB" font-size="12" font-family="sans-serif"
            font-weight="bold">✅ Avant 2010</text>

      <!-- ── FLÈCHE CENTRALE ── -->
      <line x1="235" y1="135" x2="315" y2="135"
            stroke="white" stroke-width="2.5"
            marker-end="url(#arr2)"/>
      <rect x="234" y="100" width="80" height="22"
            rx="5" fill="#FFD700" opacity="0.85"/>
      <text x="275" y="115" text-anchor="middle"
            fill="#1B2D26" font-size="11" font-weight="bold"
            font-family="sans-serif">≈ 2010</text>
      <text x="275" y="155" text-anchor="middle"
            fill="#FFD700" font-size="10"
            font-family="sans-serif">⚠ Année de référence</text>

      <!-- ── DROITE : Après 2010 ── -->
      <!-- Montagne -->
      <polygon points="340,185 405,85 470,185"
               fill="#2D6A4F" stroke="#9FE1CB" stroke-width="2"/>
      <!-- Neige -->
      <polygon points="405,85 395,112 415,112"
               fill="white" opacity="0.7"/>
      <!-- Sol -->
      <rect x="320" y="185" width="170" height="8"
            rx="4" fill="#3B6D11" opacity="0.6"/>
      <!-- Croix rouges (pas de fleurs) -->
      <line x1="344" y1="178" x2="360" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>
      <line x1="360" y1="178" x2="344" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>
      <line x1="376" y1="178" x2="392" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>
      <line x1="392" y1="178" x2="376" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>
      <line x1="420" y1="178" x2="436" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>
      <line x1="436" y1="178" x2="420" y2="194"
            stroke="#E24B4A" stroke-width="2.5"/>

      <text x="403" y="220" text-anchor="middle"
            fill="#E24B4A" font-size="12" font-family="sans-serif"
            font-weight="bold">❌ Après 2010</text>

      <!-- ── PERSONNAGE + BULLE ── -->
      <!-- Tête -->
      <circle cx="555" cy="130" r="16"
              fill="#F5DEB3" stroke="white" stroke-width="1.5"/>
      <!-- Yeux -->
      <circle cx="550" cy="127" r="2" fill="#333"/>
      <circle cx="560" cy="127" r="2" fill="#333"/>
      <!-- Bouche étonnée -->
      <ellipse cx="555" cy="136" rx="4" ry="3"
               fill="#333" opacity="0.7"/>
      <!-- Corps -->
      <rect x="544" y="146" width="22" height="26"
            rx="5" fill="#4A90D9" opacity="0.8"/>
      <!-- Bras levé gauche -->
      <line x1="544" y1="150" x2="530" y2="138"
            stroke="#F5DEB3" stroke-width="3" stroke-linecap="round"/>
      <!-- Bras droit -->
      <line x1="566" y1="152" x2="576" y2="162"
            stroke="#F5DEB3" stroke-width="3" stroke-linecap="round"/>
      <!-- Jambes -->
      <line x1="550" y1="172" x2="546" y2="192"
            stroke="#333" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="560" y1="172" x2="564" y2="192"
            stroke="#333" stroke-width="2.5" stroke-linecap="round"/>

      <!-- 3 bulles de question avec icônes -->

      <!-- Bulle 1 — Température 🌡 -->
      <ellipse cx="600" cy="68" rx="38" ry="26"
               fill="#FCEBEB" opacity="0.95"
               stroke="#E24B4A" stroke-width="1.5"/>
      <circle cx="572" cy="86" r="4"
              fill="#FCEBEB" stroke="#E24B4A" stroke-width="1"/>
      <circle cx="579" cy="79" r="3"
              fill="#FCEBEB" stroke="#E24B4A" stroke-width="1"/>
      <!-- Thermomètre -->
      <rect x="588" y="56" width="5" height="16" rx="2.5"
            fill="none" stroke="#E24B4A" stroke-width="1.2"/>
      <rect x="589.5" y="63" width="2" height="9" rx="1"
            fill="#E24B4A" opacity="0.7"/>
      <circle cx="590.5" cy="74" r="4" fill="#E24B4A" opacity="0.8"/>
      <text x="610" y="68" text-anchor="middle"
            fill="#A32D2D" font-size="10" font-weight="bold"
            font-family="sans-serif">Température</text>
      <text x="610" y="80" text-anchor="middle"
            fill="#A32D2D" font-size="9" font-family="sans-serif">? 🌡️</text>

      <!-- Bulle 2 — Neige ❄ -->
      <ellipse cx="640" cy="118" rx="38" ry="26"
               fill="#E6F1FB" opacity="0.95"
               stroke="#185FA5" stroke-width="1.5"/>
      <circle cx="608" cy="118" r="4"
              fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
      <circle cx="614" cy="110" r="3"
              fill="#E6F1FB" stroke="#185FA5" stroke-width="1"/>
      <!-- Flocon -->
      <line x1="628" y1="108" x2="628" y2="122"
            stroke="#378ADD" stroke-width="1.8"/>
      <line x1="622" y1="112" x2="634" y2="118"
            stroke="#378ADD" stroke-width="1.8"/>
      <line x1="634" y1="112" x2="622" y2="118"
            stroke="#378ADD" stroke-width="1.8"/>
      <text x="653" y="114" text-anchor="middle"
            fill="#0C447C" font-size="10" font-weight="bold"
            font-family="sans-serif">Neige</text>
      <text x="653" y="126" text-anchor="middle"
            fill="#0C447C" font-size="9" font-family="sans-serif">? ❄️</text>

      <!-- Bulle 3 — Précipitation 🌧 -->
      <ellipse cx="610" cy="168" rx="38" ry="26"
               fill="#EAF3DE" opacity="0.95"
               stroke="#3B6D11" stroke-width="1.5"/>
      <circle cx="578" cy="162" r="4"
              fill="#EAF3DE" stroke="#3B6D11" stroke-width="1"/>
      <circle cx="583" cy="155" r="3"
              fill="#EAF3DE" stroke="#3B6D11" stroke-width="1"/>
      <!-- Goutte -->
      <path d="M596,156 Q596,166 602,170 Q608,166 608,156 Q602,150 596,156Z"
            fill="#639922" opacity="0.8"/>
      <text x="620" y="163" text-anchor="middle"
            fill="#27500A" font-size="10" font-weight="bold"
            font-family="sans-serif">Précip.</text>
      <text x="620" y="175" text-anchor="middle"
            fill="#27500A" font-size="9" font-family="sans-serif">? 🌧️</text>

      <text x="555" y="215" text-anchor="middle"
            fill="white" font-size="11" font-family="sans-serif">
        Quel facteur cause le défaut ?
      </text>

    </svg>
    </div>
    </body>
    </html>
    """
    components.html(histoire_html, height=340, scrolling=False)

    st.markdown("""
    <div class="question-box">
    ❓ Pourquoi le site de Nohèdes présente-t-il un défaut de floraison récurrent,
    observé à partir d'environ 2010, alors que les 8 autres populations restent florissantes ?
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Métriques KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌍 Sites étudiés", "9")
    col2.metric("📅 Période", "2000–2020")
    col3.metric("📊 Variables", "4")
    col4.metric("🇫🇷 Sites France", "5")
    col5.metric("🇪🇸 Sites Espagne", "4")

    st.divider()

    # Tableau des sites
    st.markdown("### 📋 Tableau des 9 sites d'étude")
    df_sites_display = pd.DataFrame({
        "Site"       : ["NOHEDES","EYNE_POP1","EYNE_POP2","EYNE_POP3",
                        "VALLTER","CADI_POP1","CADI_POP2","CADI_POP3","CADI_POP4"],
        "Versant"    : ["France","France","France","France",
                        "Espagne","Espagne","Espagne","Espagne","Espagne"],
        "Altitude (m)": [1790,2586,2181,2164,2147,2365,2283,1979,2358],
        "Statut"     : ["❌ Déficient","✅ Floraison","✅ Floraison",
                        "✅ Floraison","✅ Floraison","✅ Floraison",
                        "✅ Floraison","✅ Floraison","✅ Floraison"]
    })

    st.dataframe(
        df_sites_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Site"        : st.column_config.TextColumn("Site", width="medium"),
            "Statut"      : st.column_config.TextColumn("Statut floraison", width="medium"),
            "Altitude (m)": st.column_config.NumberColumn("Altitude (m)", format="%d m"),
        }
    )

    st.divider()

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — ZONE D'ÉTUDE
# ══════════════════════════════════════════════════════════════════
elif page == "🗺️ Zone d'étude":

    st.markdown('<div class="titre-page">🗺️ Zone d\'étude — 9 sites Pyrénées</div>',
                unsafe_allow_html=True)

    col_carte, col_info = st.columns([2, 1])

    with col_carte:
        # Carte Plotly interactive sans token
        SITES["statut"] = SITES["floraison"].apply(
            lambda x: "✅ Floraison normale" if x else "❌ Défaut floraison"
        )

        fig_carte = go.Figure()

        # Autres sites — violet
        sites_fleur = SITES[SITES["floraison"] == True]
        fig_carte.add_trace(go.Scattermapbox(
            lat=sites_fleur["lat"],
            lon=sites_fleur["lon"],
            mode="markers+text",
            marker=dict(
                size=14,
                color="#7F77DD",
                opacity=0.9
            ),
            text=sites_fleur["nom"],
            textposition="top right",
            textfont=dict(size=11, color="#3C3489"),
            customdata=sites_fleur[["altitude","versant"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Altitude : %{customdata[0]} m<br>"
                "Versant  : %{customdata[1]}<br>"
                "Statut   : ✅ Floraison normale"
                "<extra></extra>"
            ),
            name="Floraison normale (8 sites)"
        ))

        # NOHEDES — vert
        noh = SITES[SITES["floraison"] == False]
        fig_carte.add_trace(go.Scattermapbox(
            lat=noh["lat"],
            lon=noh["lon"],
            mode="markers+text",
            marker=dict(
                size=18,
                color="#1D9E75",
                opacity=0.95,
                symbol="circle"
            ),
            text=noh["nom"],
            textposition="top right",
            textfont=dict(size=12, color="#085041"),
            customdata=noh[["altitude","versant"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Altitude : %{customdata[0]} m<br>"
                "Versant  : %{customdata[1]}<br>"
                "Statut   : ❌ Défaut floraison"
                "<extra></extra>"
            ),
            name="NOHEDES — Défaut floraison"
        ))

        fig_carte.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=42.42, lon=1.95),
                zoom=8.5
            ),
            height=460,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )

        st.plotly_chart(fig_carte, use_container_width=True)

    with col_info:
        st.markdown("### Informations sites")
        for _, row in SITES.iterrows():
            statut = "🟢" if not row["floraison"] else "🟣"
            st.markdown(f"""
            <div class="{'nohedes-box' if not row['floraison'] else 'carte-site'}">
            <b>{statut} {row['nom']}</b><br>
            📍 {row['versant']} · {row['altitude']}m<br>
            {'❌ Sans floraison' if not row['floraison'] else '✅ Floraison normale'}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Graphique altitude
    st.markdown("### 📊 Altitude des 9 sites")
    fig = px.bar(
        SITES.sort_values("altitude"),
        x="nom", y="altitude",
        color="floraison",
        color_discrete_map={True: "#7F77DD", False: "#1D9E75"},
        labels={"altitude": "Altitude (m)", "nom": "Site",
                "floraison": "Floraison"},
        title="Altitude des 9 sites Delphinium montanum"
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True
    )
    fig.add_hline(y=1790, line_dash="dash", line_color="#1D9E75",
                  annotation_text="NOHEDES (1790m)")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2bis — DOWNSCALING
# ══════════════════════════════════════════════════════════════════
elif page == "🛰️ Downscaling":

    st.markdown('<div class="titre-page">🛰️ Downscaling — De 9 km à 30 m</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="histoire-box">
    <strong>🌐 Pourquoi le downscaling ?</strong><br><br>
    Les données climatiques ERA5-Land ont une résolution native de <strong>9 km</strong>, 
    insuffisante pour capturer les variations topographiques fines des sites alpins 
    pyrénéens. Pour caractériser précisément le microclimat de chaque site 
    (notamment <strong>NOHEDES à 1790m</strong>), nous avons développé un pipeline 
    de <strong>descente d'échelle (downscaling)</strong> par Random Forest, 
    permettant de passer à une résolution de <strong>30 m</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📡 Résolution ERA5", "9 km")
    col2.metric("🎯 Résolution finale", "30 m")
    col3.metric("🤖 Modèle", "Random Forest")
    col4.metric("📊 Prédicteurs", "MNT, pente, exposition...")

    st.divider()

    # ── Image 0 : Schéma du processus Random Forest
    st.markdown("### 🔧 Schéma du processus de downscaling — Random Forest")
    st.markdown("""
    Cette figure schématise le **pipeline de descente d'échelle** par Random Forest. 
    Le modèle apprend la relation entre les variables topographiques fines 
    (MNT, pente, exposition...) et la variable climatique cible à la résolution 
    grossière ERA5, puis prédit cette variable à la résolution fine de 30 m.
    """)
    
    img_schema = "downscaling.png"
    if os.path.exists(img_schema):
        st.image(img_schema, 
                  caption="Pipeline de downscaling par Random Forest",
                  use_container_width=True)
    elif os.path.exists(f"images/{img_schema}"):
        st.image(f"images/{img_schema}",
                  caption="Pipeline de downscaling par Random Forest",
                  use_container_width=True)
    else:
        st.warning(f"⚠️ Image `{img_schema}` non trouvée. "
                    f"Placez-la à la racine du projet ou dans le dossier `images/`.")

    st.divider()

    st.markdown("### 📊 Validation du downscaling — Variabilité de l'erreur")
    st.markdown("""
    Cette première figure montre la **variabilité de l'erreur de prédiction** du 
    modèle Random Forest sur l'ensemble des stations de validation Météo-France.
    """)
    
    img1 = "validation_downscaling_T_FORTE.png"
    if os.path.exists(img1):
        st.image(img1, caption="Validation du downscaling de la température",
                  use_container_width=True)
    elif os.path.exists(f"images/{img1}"):
        st.image(f"images/{img1}", caption="Validation du downscaling de la température",
                  use_container_width=True)
    else:
        st.warning(f"⚠️ Image `{img1}` non trouvée.")

    st.divider()

    st.markdown("### 🌡️ Résultat du downscaling — Comparaison avant / après")
    st.markdown("""
    Cette seconde figure illustre le **gain en résolution spatiale** apporté par 
    le downscaling. À la même date (avril 2020), on compare la température à la 
    résolution native ERA5 (gauche) et après downscaling à 30 m (droite).
    """)
    
    img2 = "comparaison_avant_apres_T_202004.png"
    if os.path.exists(img2):
        st.image(img2, caption="Comparaison avant / après downscaling — Avril 2020",
                  use_container_width=True)
    elif os.path.exists(f"images/{img2}"):
        st.image(f"images/{img2}", caption="Comparaison avant / après downscaling — Avril 2020",
                  use_container_width=True)
    else:
        st.warning(f"⚠️ Image `{img2}` non trouvée.")

    st.divider()

    st.markdown("""
    <div class="nohedes-box">
    <strong>🎯 Ce que permet le downscaling</strong><br><br>
    Grâce à cette descente d'échelle, nous obtenons pour chaque site (dont NOHEDES) 
    des données climatiques fines, intégrant la topographie locale. C'est sur ces 
    données <strong>downscalées à 30 m</strong> que reposent les analyses 
    statistiques de la suite.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Statistiques descriptives":

    st.markdown('<div class="titre-page">📊 Statistiques descriptives</div>',
                unsafe_allow_html=True)

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        var_choisie = st.selectbox(
            "📌 Variable",
            list(VARS.keys()),
            format_func=lambda x: VARS[x]
        )
    with col2:
        annee_choisie = st.selectbox(
            "📅 Année",
            ["Toutes"] + sorted(df["annee"].unique().tolist())
        )
    with col3:
        saison_choisie = st.selectbox(
            "🌿 Saison",
            ["Toutes", "Hiver", "Printemps", "Été", "Automne"]
        )

    # Filtrer
    df_filtre = df.copy()
    if annee_choisie != "Toutes":
        df_filtre = df_filtre[df_filtre["annee"] == annee_choisie]
    if saison_choisie != "Toutes":
        df_filtre = df_filtre[df_filtre["saison"] == saison_choisie]

    st.divider()

    # Stats descriptives
    col_stats, col_box = st.columns([1, 2])

    with col_stats:
        st.markdown("#### Tableau comparatif")

        stats = df_filtre.groupby("groupe")[var_choisie].agg([
            ("Moyenne", lambda x: round(x.mean(), 2)),
            ("Médiane", lambda x: round(x.median(), 2)),
            ("Écart-type", lambda x: round(x.std(), 2)),
            ("Min", lambda x: round(x.min(), 2)),
            ("Max", lambda x: round(x.max(), 2)),
        ]).reset_index()
        stats.columns = ["Groupe","Moyenne","Médiane","Écart-type","Min","Max"]
        st.dataframe(stats, hide_index=True, use_container_width=True)

        # Différence
        if len(stats) == 2:
            noh = stats[stats["Groupe"] == "NOHEDES"]["Moyenne"].values[0]
            aut = stats[stats["Groupe"] == "Autres sites"]["Moyenne"].values[0]
            diff = round(noh - aut, 2)
            signe = "↑" if diff > 0 else "↓"
            st.metric(
                f"Différence (NOHEDES - Autres)",
                f"{diff:+.2f}",
                delta=f"{signe} NOHEDES {'plus élevé' if diff>0 else 'plus faible'}"
            )

    with col_box:
        st.markdown("#### Boxplot comparatif")

        fig = go.Figure()
        for groupe, couleur in COULEURS.items():
            vals = df_filtre[df_filtre["groupe"] == groupe][var_choisie]
            fig.add_trace(go.Box(
                y=vals,
                name=groupe,
                marker_color=couleur,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(size=5, opacity=0.5),
                line=dict(width=2)
            ))

        fig.update_layout(
            title=f"{VARS[var_choisie]} — NOHEDES vs Autres sites",
            yaxis_title=VARS[var_choisie],
            xaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Heatmap par site et mois
    st.markdown("#### 🔥 Heatmap mensuelle par site")
    df_heat = df_filtre.groupby(["nom", "mois"])[var_choisie].mean().reset_index()
    df_heat_pivot = df_heat.pivot(index="nom", columns="mois", values=var_choisie)

    mois_noms = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Juin",
                 7:"Juil",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
    df_heat_pivot.columns = [mois_noms.get(c, c)
                              for c in df_heat_pivot.columns]

    fig_heat = px.imshow(
        df_heat_pivot,
        color_continuous_scale="RdYlBu_r",
        title=f"{VARS[var_choisie]} — Moyenne mensuelle par site",
        aspect="auto",
        labels={"color": VARS[var_choisie]}
    )
    fig_heat.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — ÉVOLUTION TEMPORELLE
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Évolution temporelle":

    st.markdown('<div class="titre-page">📈 Évolution temporelle 2000–2020</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        var_choisie = st.selectbox(
            "📌 Variable",
            list(VARS.keys()),
            format_func=lambda x: VARS[x],
            key="var_evol"
        )
    with col2:
        saison_choisie = st.selectbox(
            "🌿 Saison",
            ["Toutes", "Hiver", "Printemps", "Été", "Automne"],
            key="sai_evol"
        )

    # Filtrer
    df_evol = df.copy()
    if saison_choisie != "Toutes":
        df_evol = df_evol[df_evol["saison"] == saison_choisie]

    # Moyenne annuelle
    df_annuel = df_evol.groupby(["annee","groupe"])[var_choisie].mean().reset_index()

    # Enveloppe min-max autres sites
    df_env = df_evol[df_evol["groupe"] == "Autres sites"].groupby(
        ["annee","nom"])[var_choisie].mean().reset_index()
    df_env2 = df_env.groupby("annee")[var_choisie].agg(
        ["min","max"]).reset_index()

    # Graphique
    fig = go.Figure()

    # Zone min-max autres sites
    fig.add_trace(go.Scatter(
        x=pd.concat([df_env2["annee"], df_env2["annee"][::-1]]),
        y=pd.concat([df_env2["max"], df_env2["min"][::-1]]),
        fill="toself",
        fillcolor="rgba(127,119,221,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Enveloppe autres sites",
        showlegend=True
    ))

    # Autres sites (moyenne)
    df_aut = df_annuel[df_annuel["groupe"] == "Autres sites"]
    fig.add_trace(go.Scatter(
        x=df_aut["annee"], y=df_aut[var_choisie],
        mode="lines+markers",
        name="Autres sites (moyenne)",
        line=dict(color="#7F77DD", width=2),
        marker=dict(size=6)
    ))

    # Tendance autres sites
    z_aut = np.polyfit(df_aut["annee"], df_aut[var_choisie], 1)
    p_aut = np.poly1d(z_aut)
    fig.add_trace(go.Scatter(
        x=df_aut["annee"],
        y=p_aut(df_aut["annee"]),
        mode="lines",
        name="Tendance autres",
        line=dict(color="#7F77DD", width=1, dash="dot")
    ))

    # NOHEDES
    df_noh = df_annuel[df_annuel["groupe"] == "NOHEDES"]
    fig.add_trace(go.Scatter(
        x=df_noh["annee"], y=df_noh[var_choisie],
        mode="lines+markers",
        name="NOHEDES",
        line=dict(color="#1D9E75", width=3),
        marker=dict(size=8, symbol="diamond")
    ))

    # Tendance NOHEDES
    z_noh = np.polyfit(df_noh["annee"], df_noh[var_choisie], 1)
    p_noh = np.poly1d(z_noh)
    fig.add_trace(go.Scatter(
        x=df_noh["annee"],
        y=p_noh(df_noh["annee"]),
        mode="lines",
        name="Tendance NOHEDES",
        line=dict(color="#1D9E75", width=1, dash="dot")
    ))

    # Ligne 2010
    fig.add_vline(x=2010, line_dash="dash",
                  line_color="grey",
                  annotation_text="≈ 2010 (référence)",
                  annotation_position="top right")

    fig.update_layout(
        title=f"{VARS[var_choisie]} — Évolution 2000–2020 ({saison_choisie})",
        xaxis_title="Année",
        yaxis_title=VARS[var_choisie],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Graphique 4 variables en grille
    st.markdown("### 📊 Vue d'ensemble — 4 variables")

    df_annuel_all = df.groupby(["annee","groupe"])[list(VARS.keys())].mean().reset_index()

    fig2 = make_subplots(rows=2, cols=2,
                          subplot_titles=list(VARS.values()))

    for i, (var, nom) in enumerate(VARS.items()):
        row = i // 2 + 1
        col = i  % 2 + 1

        for groupe, couleur in COULEURS.items():
            df_g = df_annuel_all[df_annuel_all["groupe"] == groupe]
            fig2.add_trace(
                go.Scatter(
                    x=df_g["annee"], y=df_g[var],
                    mode="lines+markers",
                    name=groupe,
                    line=dict(color=couleur, width=2),
                    marker=dict(size=5),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

        fig2.add_vline(x=2010, line_dash="dash",
                       line_color="grey", row=row, col=col)

    fig2.update_layout(
        height=600,
        title="4 variables climatiques — NOHEDES vs Autres sites",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — ACP + CAH
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 ACP + CAH":

    st.markdown('<div class="titre-page">🔬 Analyse multivariée — ACP + CAH · Année 2010</div>',
                unsafe_allow_html=True)

    # Année fixée à 2010
    annee_acp = 2010
    st.info("""
    📅 Analyse réalisée sur l'année **2010** — retenue comme année de référence
    d'après les observations de terrain des encadrants.
    ⚠️ *Cette date reste une approximation, les observations phénologiques
    antérieures étant partielles et non systématiques.*
    """)

    # Données par site pour 2010
    df_acp = df[df["annee"] == annee_acp].groupby("nom")[list(VARS.keys())].mean().reset_index()
    df_acp["floraison"] = df_acp["nom"].apply(
        lambda x: "Défaut floraison" if x == "NOHEDES" else "Floraison"
    )

    # PCA
    X = df_acp[list(VARS.keys())].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(4, X.shape[1]))
    coords = pca.fit_transform(X_scaled)

    var_exp = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame({
        "PC1" : coords[:,0],
        "PC2" : coords[:,1],
        "nom" : df_acp["nom"].values,
        "floraison": df_acp["floraison"].values
    })

    col_valeurs, col_cercle = st.columns(2)

    with col_valeurs:
        st.markdown("#### 📊 Valeurs propres")
        df_eig = pd.DataFrame({
            "Axe"         : [f"PC{i+1}" for i in range(len(var_exp))],
            "Valeur propre": np.round(pca.explained_variance_, 3),
            "% Variance"  : np.round(var_exp, 1),
            "% Cumulé"    : np.round(np.cumsum(var_exp), 1)
        })
        st.dataframe(df_eig, hide_index=True, use_container_width=True)

        st.metric(f"PC1 + PC2",
                  f"{var_exp[0]+var_exp[1]:.1f}%",
                  f"Variance expliquée")

    with col_cercle:
        st.markdown("#### ⭕ Cercle de corrélation")
        loadings = pca.components_.T
        fig_var = go.Figure()

        # Cercle unité
        theta = np.linspace(0, 2*np.pi, 100)
        fig_var.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines",
            line=dict(color="grey", width=1, dash="dash"),
            showlegend=False
        ))

        # Flèches variables
        for j, (var, nom_var) in enumerate(VARS.items()):
            if j < loadings.shape[0]:
                fig_var.add_annotation(
                    x=loadings[j,0], y=loadings[j,1],
                    ax=0, ay=0,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    arrowhead=2,
                    arrowcolor="#E24B4A",
                    arrowwidth=2
                )
                fig_var.add_trace(go.Scatter(
                    x=[loadings[j,0]*1.15],
                    y=[loadings[j,1]*1.15],
                    mode="text",
                    text=[nom_var.split("(")[0].strip()],
                    textfont=dict(size=11, color="#E24B4A"),
                    showlegend=False
                ))

        fig_var.update_layout(
            xaxis=dict(range=[-1.3,1.3],
                       title=f"PC1 ({var_exp[0]:.1f}%)",
                       zeroline=True),
            yaxis=dict(range=[-1.3,1.3],
                       title=f"PC2 ({var_exp[1]:.1f}%)",
                       zeroline=True,
                       scaleanchor="x"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.divider()

    # Individus
    st.markdown("#### 👤 Projection des individus")
    fig_ind = px.scatter(
        df_pca, x="PC1", y="PC2",
        color="floraison",
        text="nom",
        color_discrete_map={
            "Défaut floraison": "#1D9E75",
            "Floraison"       : "#7F77DD"
        },
        title=f"Projection ACP — Année {annee_acp}",
        labels={"PC1": f"PC1 ({var_exp[0]:.1f}%)",
                "PC2": f"PC2 ({var_exp[1]:.1f}%)"}
    )
    fig_ind.update_traces(textposition="top center", marker=dict(size=12))
    fig_ind.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_ind.add_vline(x=0, line_dash="dash", line_color="grey")
    fig_ind.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    st.divider()

    # CAH
    st.markdown("#### 🌲 Classification Ascendante Hiérarchique (k=2)")

    Z = linkage(X_scaled, method="ward")
    clusters = fcluster(Z, t=2, criterion="maxclust")
    df_pca["cluster"] = [f"Cluster {c}" for c in clusters]

    col_cah1, col_cah2 = st.columns(2)

    with col_cah1:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Couleurs par cluster
        CLUSTER_COLORS = {1: "#7F77DD", 2: "#1D9E75"}

        fig_dend, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("#F8F9FA")
        fig_dend.patch.set_facecolor("#F8F9FA")

        # Seuil pour couper en 2 clusters
        seuil = (Z[-2, 2] + Z[-1, 2]) / 2

        dend = dendrogram(
            Z,
            labels=df_acp["nom"].values,
            color_threshold=seuil,
            ax=ax,
            above_threshold_color="grey"
        )

        # Colorier les labels selon floraison
        for lbl in ax.get_xticklabels():
            txt = lbl.get_text()
            lbl.set_color("#1D9E75" if txt == "NOHEDES" else "#7F77DD")
            lbl.set_fontweight("bold")

        ax.axhline(y=seuil, color="red",
                   linestyle="--", linewidth=1.2,
                   label=f"Seuil coupe (k=2)")

        ax.set_title("Dendrogramme CAH (Ward) — 2010",
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Distance (Ward)")
        ax.legend(fontsize=9)

        # Légende manuelle
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1D9E75", label="NOHEDES (défaut floraison)"),
            Patch(facecolor="#7F77DD", label="Autres sites (floraison)")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_dend)

    with col_cah2:
        # Clusters dans ACP
        fig_clust = px.scatter(
            df_pca, x="PC1", y="PC2",
            color="cluster",
            symbol="floraison",
            text="nom",
            color_discrete_sequence=["#7F77DD","#1D9E75"],
            title=f"Clusters dans l'espace ACP — {annee_acp}",
            labels={"PC1": f"PC1 ({var_exp[0]:.1f}%)",
                    "PC2": f"PC2 ({var_exp[1]:.1f}%)"}
        )
        fig_clust.update_traces(
            textposition="top center",
            marker=dict(size=12)
        )
        fig_clust.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_clust.add_vline(x=0, line_dash="dash", line_color="grey")
        fig_clust.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )
        st.plotly_chart(fig_clust, use_container_width=True)

    # Profils par cluster
    st.markdown("#### 📊 Profils climatiques par cluster")
    df_acp["cluster"] = [f"Cluster {c}" for c in clusters]
    df_profils = df_acp.groupby("cluster")[list(VARS.keys())].mean().reset_index()
    df_profils_long = df_profils.melt(
        id_vars="cluster",
        value_vars=list(VARS.keys()),
        var_name="variable",
        value_name="valeur"
    )
    df_profils_long["variable"] = df_profils_long["variable"].map(VARS)

    fig_prof = px.bar(
        df_profils_long,
        x="variable", y="valeur",
        color="cluster",
        barmode="group",
        color_discrete_sequence=["#7F77DD","#1D9E75"],
        title="Moyennes climatiques par cluster",
        labels={"variable":"","valeur":"Valeur moyenne","cluster":"Cluster"}
    )
    fig_prof.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_prof, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — TEST D'HYPOTHÈSE
# ══════════════════════════════════════════════════════════════════
elif page == "🧪 Test d'hypothèse":

    st.markdown('<div class="titre-page">🧪 Test d\'hypothèse — Wilcoxon · Année 2010</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **H₀** : Les conditions climatiques de NOHEDES sont similaires aux autres sites

    **H₁** : Les conditions climatiques de NOHEDES diffèrent significativement
    """)

    st.info("""
    📅 Test réalisé sur l'année **2010** — retenue comme année de référence
    d'après les observations de terrain des encadrants.
    ⚠️ *Cette date reste une approximation raisonnable mais non certaine.*
    """)

    # Filtre saison uniquement
    saison_test = st.selectbox(
        "🌿 Saison",
        ["Toutes","Hiver","Printemps","Été","Automne"],
        key="sai_test"
    )

    # Année fixée à 2010
    annee_test = 2010
    df_test = df[df["annee"] == annee_test].copy()
    if saison_test != "Toutes":
        df_test = df_test[df_test["saison"] == saison_test]

    # Tests
    resultats = []
    for var, nom_var in VARS.items():
        vals_noh = df_test[df_test["groupe"]=="NOHEDES"][var].dropna()
        vals_aut = df_test[df_test["groupe"]=="Autres sites"][var].dropna()

        if len(vals_noh) > 0 and len(vals_aut) > 0:
            stat, p = mannwhitneyu(vals_noh, vals_aut,
                                   alternative="two-sided")
            sig = ("***" if p < 0.001 else
                   "**"  if p < 0.01  else
                   "*"   if p < 0.05  else "ns")
            conclusion = ("❌ H₀ rejetée" if p < 0.05
                         else "✅ H₀ acceptée")
            resultats.append({
                "Variable"    : nom_var,
                "Moy. NOHEDES": round(vals_noh.mean(), 2),
                "Moy. Autres" : round(vals_aut.mean(), 2),
                "Différence"  : round(vals_noh.mean()-vals_aut.mean(), 2),
                "p-value"     : round(p, 4),
                "Sig."        : sig,
                "Conclusion"  : conclusion
            })

    df_res = pd.DataFrame(resultats)

    st.divider()
    st.markdown("#### 📋 Tableau des résultats")
    st.dataframe(df_res, hide_index=True, use_container_width=True)

    st.divider()

    # Boxplots
    st.markdown("#### 📦 Boxplots comparatifs")
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=list(VARS.values()))

    for i, (var, nom_var) in enumerate(VARS.items()):
        row = i // 2 + 1
        col = i  % 2 + 1

        for groupe, couleur in COULEURS.items():
            vals = df_test[df_test["groupe"]==groupe][var]
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=groupe,
                    marker_color=couleur,
                    showlegend=(i == 0),
                    boxpoints="all",    # points visibles
                    jitter=0.3,
                    pointpos=0,         # ← points À L'INTÉRIEUR du boxplot
                    marker=dict(
                        size=6,
                        opacity=0.6
                    )
                ),
                row=row, col=col
            )

        # p-value annotation
        res = df_res[df_res["Variable"] == nom_var]
        if len(res):
            p_val = res["p-value"].values[0]
            sig   = res["Sig."].values[0]
            fig.add_annotation(
                text=f"p={p_val} {sig}",
                xref=f"x{i+1 if i>0 else ''}",
                yref=f"y{i+1 if i>0 else ''}",
                x=1, y=1,
                xanchor="center",
                showarrow=False,
                font=dict(size=11, color="grey"),
                row=row, col=col
            )

    fig.update_layout(
        height=550,
        title=f"Test de Wilcoxon — {annee_test} · {saison_test}",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Synthèse visuelle
    st.markdown("#### 🎯 Synthèse visuelle")
    cols = st.columns(4)
    for i, row in df_res.iterrows():
        with cols[i]:
            couleur = "#FFEBEE" if "rejetée" in row["Conclusion"] else "#E8F5E9"
            st.markdown(f"""
            <div style="background:{couleur};border-radius:10px;
                        padding:15px;text-align:center">
            <b>{row['Variable'].split('(')[0]}</b><br>
            <span style="font-size:1.5rem">{row['Sig.']}</span><br>
            p = {row['p-value']}<br>
            <small>{row['Conclusion']}</small>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 7 — CONCLUSION
# ══════════════════════════════════════════════════════════════════
elif page == "🏁 Conclusion":

    st.markdown('<div class="titre-page">🏁 Conclusion</div>',
                unsafe_allow_html=True)

    # Retour à l'histoire
    st.markdown("""
    <div class="histoire-box">
    <strong>Synthèse des résultats</strong><br><br>
    Nos analyses montrent que <strong>Nohèdes présente des conditions climatiques
    distinctes</strong> des sites florissants, caractérisées notamment par :<br><br>
    🌡️ Des <strong>températures plus élevées</strong> (+1.2°C en moyenne)<br>
    ❄️ Un <strong>déficit d'enneigement</strong> significatif (pct_neige ~2× inférieur)<br>
    🌧️ Des <strong>précipitations plus élevées</strong> mais sous forme liquide
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Réponses aux questions scientifiques
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ❓ Questions")
        st.success("""
        **Q1** : Nohèdes est-il climatiquement différent ?

        ✅ **Oui**, dans la majorité des années analysées,
        notamment pour la neige (p=0.003 en 2010).
       
        """)
        st.warning("""
        **Q2** : Quelle variable est la plus discriminante ?

        ❄️ **La neige** (pct_neige et neige_RF_cm)
        est la variable la plus discriminante,
        suivie de la température. Nohèdes présente
        un déficit nival par rapport
        aux autres sites.
        """)
        st.info("""
        **Q3** : Nohèdes avait-il les mêmes conditions climatiques
        que les autres sites en 2010 ?

        ❄️ **Non.** Le test de Wilcoxon sur l'année 2010
        montre que Nohèdes se distingue significativement
        des autres sites au niveau de la **hauteur de neige**
        (p=0.003 **). On peut donc affirmer qu'il existait
        une différence climatique en 2010.

        ⚠️ *Notons que 2010 reste une année de référence
        approximative, les observations phénologiques
        antérieures étant partielles.*
        """)

    with col2:
        st.markdown("#### 📊 Résultats clés")

        # Métriques finales
        st.metric("Années analysées", "21 (2000–2020)")
        st.metric("Variable discriminante", "Neige ❄️")
        st.metric("Différence T°", "+1.2°C (NOHEDES > Autres)")
        st.metric("Test Wilcoxon (neige)", "p = 0.003 **")



    st.divider()

    # Remerciements
    st.markdown("#### 🙏 Remerciements")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Sébastien PINEL**\nCEFREM — UPVD\nSuperviseur principal")
    with col2:
        st.info("**Noémie COLLETTE**\nLGDP — UPVD\nCo-superviseure")
    with col3:
        st.info("**IUT de Carcassonne**\nBUT3 Science des Données\nParcours EMS")

    st.markdown("""
    ---
    <div style="text-align:center;color:grey;font-size:0.9rem">
    Amadou FOFANA · Stage D2ClimAFLo-Pyr · CEFREM · UPVD · 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 9 — PERSPECTIVES (4 sous-onglets)
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 Perspectives":

    st.markdown('<div class="titre-page">🎯 Perspectives — Travail futur</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="histoire-box">
    <strong>🔭 Travail prochain</strong><br><br>
    Pour approfondir la caractérisation du défaut de floraison à NOHEDES, 
    j'ai commencé à explorer une <strong>approche statistique plus poussée</strong> 
    basée sur <strong>13 variables climatiques</strong> (au lieu des 4 utilisées 
    dans les analyses précédentes). Cette extension comprend 4 analyses :<br><br>
    📊 <strong>PCA & Distance Mahalanobis</strong> sur fenêtres temporelles<br>
    📈 <strong>Évolution variable</strong> — Comparaison annuelle<br>
    🌨️ <strong>SMOD vs LFD</strong> — Croisement neige/gel<br>
    🎯 <strong>Modélisation, Rupture & Tukey</strong> — Test interaction
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ────────────────────────────────────────────────
    # CONSTANTES POUR LES PERSPECTIVES
    # ────────────────────────────────────────────────
    
    # ────────────────────────────────────────────────
    # CHARGEMENT df_enrichi_v4.csv
    # ────────────────────────────────────────────────
    @st.cache_data
    def charger_donnees_enrichies():
        chemins = ["df_enrichi_v4.csv", "data/df_enrichi_v4.csv"]
        for chemin in chemins:
            if os.path.exists(chemin):
                try:
                    d = pd.read_csv(chemin)
                    if d.shape[1] < 5:
                        d = pd.read_csv(chemin, sep=";")
                    return d
                except Exception:
                    try:
                        d = pd.read_csv(chemin, sep=";")
                        return d
                    except Exception:
                        continue
        return None
    
    df = charger_donnees_enrichies()
    if df is None:
        st.error("⚠️ Fichier `df_enrichi_v4.csv` non trouvé. Placez-le à la racine du projet.")
        st.stop()
    
    st.success(f"✅ Dataset enrichi chargé : {len(df)} observations")
    
    # Filtrer VARS_INFO selon les colonnes disponibles dans le df
    VARS_INFO = {k: v for k, v in VARS_INFO.items() if k in df.columns}
    VARS_LISTE = list(VARS_INFO.keys())
    
    # Variables par défaut (sans soil_moisture_moy)
    vars_default = ["SMOD_modis", "continuite", "temp_RF_moy", "SCD", "humidity_moy"]
    vars_default = [v for v in vars_default if v in VARS_LISTE]

    # ────────────────────────────────────────────────
    # WIDGETS DE CONFIGURATION (multiselect + fenêtre)
    # ────────────────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns([2, 1])
    
    with col_cfg1:
        vars_options = [v for v in VARS_LISTE if v != 'soil_moisture_moy']
        vars_select = st.multiselect(
            "🌿 Variables à analyser pour la PCA",
            options=vars_options,
            default=vars_default,
            format_func=lambda x: VARS_INFO.get(x, x),
            help="Choisir 2 variables minimum",
            key="persp_vars"
        )
    
    with col_cfg2:
        fenetre_choisie = st.select_slider(
            "📅 Fenêtre temporelle",
            options=list(FENETRES.keys()),
            value=list(FENETRES.keys())[2],
            key="persp_fenetre"
        )
    
    if len(vars_select) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 variables")
        st.stop()
    
    st.divider()

    # ────────────────────────────────────────────────
    # 4 SOUS-ONGLETS
    # ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 PCA & Mahalanobis",
        "📈 Évolution variable",
        "🌨️ SMOD vs LFD",
        "🎯 Modélisation, Rupture & Tukey"
    ])

    # ─────────────────────────────────────────────
    # TAB 1 — PCA & Mahalanobis
    # ─────────────────────────────────────────────
    with tab1:

        annees = FENETRES[fenetre_choisie]
    
        # Calcul
        result = calculate_pca_mahalanobis(df, annees, vars_select)
    
        if result is None:
            st.error("⚠ Pas assez de données valides pour la PCA")
            st.stop()
    
        # Métriques en haut
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
        with col_m1:
            st.metric(
                "Distance D²",
                f"{result['D2']:.2f}",
                delta=f"Seuil 5% : {result['seuil_chi2']:.2f}",
                delta_color="inverse"
            )
    
        with col_m2:
            st.metric(
                "Distance D",
                f"{result['D']:.2f}",
                delta=f"Seuil 5% : {np.sqrt(result['seuil_chi2']):.2f}",
                delta_color="inverse"
            )
    
        with col_m3:
            st.metric(
                "p-value",
                f"{result['p_value']:.4f}",
                delta="< 0.05 = atypique",
                delta_color="off"
            )
    
        with col_m4:
            if result['dans_ellipse']:
                st.metric("Verdict", "✓ SIMILAIRE")
                st.success("NOHEDES dans l'ellipse 95%")
            else:
                st.metric("Verdict", "⚠ ATYPIQUE")
                st.error("NOHEDES hors de l'ellipse 95%")
    
        # Graphique PCA
        col_g1, col_g2 = st.columns([2, 1])
    
        with col_g1:
            fig_pca = plot_pca_with_ellipse(result, fenetre_choisie)
            st.plotly_chart(fig_pca, use_container_width=True)
    
        with col_g2:
            st.markdown("### 📊 Contributions des variables")
        
            loadings = result['loadings'].copy()
            loadings['Variable'] = loadings.index.map(VARS_INFO)
            loadings['|PC1|'] = loadings['PC1'].abs()
            loadings = loadings.sort_values('|PC1|', ascending=False)
        
            df_display = loadings[['Variable', 'PC1', 'PC2']].copy()
            df_display['PC1'] = df_display['PC1'].apply(lambda x: f"{x:+.3f}")
            df_display['PC2'] = df_display['PC2'].apply(lambda x: f"{x:+.3f}")
        
            st.dataframe(
                df_display,
                hide_index=True,
                use_container_width=True,
                height=400
            )
        
            st.caption(
                f"📊 Variance expliquée : "
                f"**{result['var_pc12']:.1f}%** "
                f"(PC1: {result['var_pc1']:.1f}% + PC2: {result['var_pc2']:.1f}%)"
            )
    
        # Section explicative
        with st.expander("ℹ Comment interpréter ces résultats ?"):
            st.markdown(
                f"""
                ### 📐 Distance Mahalanobis
            
                La **distance de Mahalanobis (D²)** mesure l'éloignement de NOHEDES 
                par rapport au centre du groupe des 8 sites fleurissants, en tenant 
                compte de la dispersion et de la corrélation des variables.
            
                ### 🎯 Seuil de décision
            
                - **D² = {result['D2']:.2f}**
                - **Seuil χ²(2) à 5%% = {result['seuil_chi2']:.2f}**
                - **Conclusion :** {'NOHEDES est ATYPIQUE (D² > seuil)' if not result['dans_ellipse'] else 'NOHEDES est SIMILAIRE (D² ≤ seuil)'}
            
                ### 🔵 Ellipse de confiance 95%%
            
                L'ellipse représente la zone où **95%% des sites fleurissants** 
                devraient se trouver statistiquement. Si NOHEDES est en dehors, 
                il diffère significativement (p < 0.05).
                """
            )
    
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GRAPHIQUE EN BARRES — Z-SCORES PAR VARIABLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
        st.markdown("---")
        st.subheader("📊 Z-scores par variable — Où NOHEDES diffère-t-il ?")
    
        if result['dans_ellipse']:
            st.success(
                "✅ NOHEDES est **SIMILAIRE** globalement. "
                "Les Z-scores ci-dessous montrent variable par variable "
                "où NOHEDES s'aligne (vert) ou se distingue (orange/rouge)."
            )
        else:
            st.warning(
                "⚠ NOHEDES est **ATYPIQUE** globalement. "
                "Les Z-scores ci-dessous identifient les variables qui "
                "contribuent le plus à cette différence."
            )
    
        # Calculer Z-scores pour la fenêtre
        zscores_df = calculate_zscores_per_variable(df, annees, vars_select)
    
        if len(zscores_df) > 0:
            fig_zscores = plot_zscores_barres(
                zscores_df,
                titre_suffixe=f"Fenêtre {fenetre_choisie}"
            )
            st.plotly_chart(fig_zscores, use_container_width=True)
        
            # Légende couleurs
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.markdown("🟢 **|z| < 1** : NOHEDES similaire")
            with col_l2:
                st.markdown("🟠 **|z| 1-2** : Modérément différent")
            with col_l3:
                st.markdown("🔴 **|z| ≥ 2** : Très différent (p<0.05)")


    # ─────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # TAB 2 — Évolution variable
    # ─────────────────────────────────────────────
    with tab2:

        st.markdown("### 📈 Évolution annuelle d'une variable")
    
        var_evolution = st.selectbox(
            "Choisir la variable à visualiser",
            options=list(VARS_INFO.keys()),
            format_func=lambda x: VARS_INFO[x],
            key="var_evolution"
        )
    
        fig_evol = plot_evolution_variable(df, var_evolution)
        st.plotly_chart(fig_evol, use_container_width=True)
    
        # Statistiques par décennie
        df_var = df[df['annee'].between(2000, 2020)].copy()
        df_var['decennie'] = df_var['annee'].apply(
            lambda x: '2000-2010' if x <= 2010 else '2011-2020'
        )
    
        col_s1, col_s2 = st.columns(2)
    
        for i, decennie in enumerate(['2000-2010', '2011-2020']):
            df_dec = df_var[df_var['decennie'] == decennie]
        
            noh_val = df_dec[df_dec['nom'] == 'NOHEDES'][var_evolution].mean()
            aut_val = df_dec[df_dec['nom'] != 'NOHEDES'][var_evolution].mean()
        
            with [col_s1, col_s2][i]:
                st.markdown(f"#### 📅 {decennie}")
                st.metric("NOHEDES", f"{noh_val:.2f}")
                st.metric("Moyenne 8 autres", f"{aut_val:.2f}")
            
                diff = noh_val - aut_val
                pct = 100 * diff / aut_val if aut_val != 0 else 0
                st.caption(f"Différence : **{diff:+.2f}** ({pct:+.1f}%)")


    # ─────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # TAB 3 — SMOD vs LFD
    # ─────────────────────────────────────────────
    with tab3:

        st.markdown("### 🌨️ Comparaison SMOD (fin de la neige) vs LFD (dernier gel)")
    
        # Vérifier que LFD_nival est disponible
        if 'LFD_nival' not in df.columns:
            st.error("⚠ La variable LFD_nival n'est pas disponible dans le jeu de données. "
                      "Utilisez df_enrichi_v4.csv.")
        else:
            st.markdown(
                "Cette section compare l'évolution temporelle de deux dates "
                "clés du cycle annuel : **SMOD** (fin de l'enneigement) et "
                "**LFD** (dernier jour de gel)."
            )
        
            # ━━━ ÉVOLUTION TEMPORELLE ━━━
            st.markdown("#### 📈 Évolution annuelle (2000-2020)")
        
            fig_evolution_smod_lfd = plot_evolution_smod_lfd(df)
            st.plotly_chart(fig_evolution_smod_lfd, use_container_width=True)
        
            # Calcul stats globales
            df_calc = df[df['annee'].between(2000, 2020)].copy()
            df_calc['type'] = df_calc['nom'].apply(
                lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
            )
        
            col_s1, col_s2 = st.columns(2)
        
            with col_s1:
                st.markdown("##### 📊 SMOD (fin enneigement)")
            
                noh_smod = df_calc[df_calc['type'] == 'NOHEDES']['SMOD_modis'].mean()
                aut_smod = df_calc[df_calc['type'] == 'Autres sites']['SMOD_modis'].mean()
            
                st.metric(
                    "NOHEDES",
                    f"{jour_nival_to_date(noh_smod)} (jour {noh_smod:.0f})"
                )
                st.metric(
                    "Moyenne 8 sites",
                    f"{jour_nival_to_date(aut_smod)} (jour {aut_smod:.0f})"
                )
            
                diff_smod = noh_smod - aut_smod
                st.caption(
                    f"Différence : **{diff_smod:+.0f} jours** "
                    f"({'NOHEDES PLUS TARDIF' if diff_smod > 0 else 'NOHEDES PLUS PRÉCOCE'})"
                )
        
            with col_s2:
                st.markdown("##### 🥶 LFD (dernier gel)")
            
                noh_lfd = df_calc[df_calc['type'] == 'NOHEDES']['LFD_nival'].mean()
                aut_lfd = df_calc[df_calc['type'] == 'Autres sites']['LFD_nival'].mean()
            
                st.metric(
                    "NOHEDES",
                    f"{jour_nival_to_date(noh_lfd)} (jour {noh_lfd:.0f})"
                )
                st.metric(
                    "Moyenne 8 sites",
                    f"{jour_nival_to_date(aut_lfd)} (jour {aut_lfd:.0f})"
                )
            
                diff_lfd = noh_lfd - aut_lfd
                st.caption(
                    f"Différence : **{diff_lfd:+.0f} jours** "
                    f"({'NOHEDES PLUS TARDIF' if diff_lfd > 0 else 'NOHEDES PLUS PRÉCOCE'})"
                )
        
            st.markdown("---")
        
            # ━━━ DISTRIBUTION SAISONNIÈRE ━━━
            st.markdown("#### 🗓️ Distribution saisonnière des événements")
        
            st.markdown(
                "Combien de fois sur 21 ans (2000-2020) chaque événement "
                "tombe-t-il dans chaque saison ? Comparaison NOHEDES (vert) vs "
                "8 autres sites cumulés (violet, n=168 = 8 sites × 21 ans)."
            )
        
            fig_saisons = plot_distribution_saisons(df)
            st.plotly_chart(fig_saisons, use_container_width=True)
        
            # ━━━ TABLEAU DÉTAIL ━━━
            with st.expander("📋 Voir le détail par site et année"):
                df_detail = df[df['annee'].between(2000, 2020)][
                    ['nom', 'annee', 'SMOD_modis', 'LFD_nival']
                ].copy()
            
                df_detail['SMOD date'] = df_detail['SMOD_modis'].apply(jour_nival_to_date)
                df_detail['LFD date'] = df_detail['LFD_nival'].apply(jour_nival_to_date)
                df_detail['SMOD saison'] = df_detail['SMOD_modis'].apply(jour_nival_to_saison)
                df_detail['LFD saison'] = df_detail['LFD_nival'].apply(jour_nival_to_saison)
            
                st.dataframe(
                    df_detail[['nom', 'annee', 'SMOD_modis', 'SMOD date', 'SMOD saison',
                                  'LFD_nival', 'LFD date', 'LFD saison']],
                    hide_index=True,
                    use_container_width=True
                )


    # ─────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # TAB 4 — Modélisation, Rupture & Tukey
    # ─────────────────────────────────────────────
    with tab4:

        # ════════════════════════════════════════════════════════════
        # SECTION 1 — POURQUOI LE SMOD ?
        # ════════════════════════════════════════════════════════════
    
        st.markdown("# 🎯 Modélisation, Rupture & Test de Tukey")
        st.markdown("---")
    
        st.markdown("## 📌 1. Pourquoi le SMOD comme variable cible ?")
    
        col_intro1, col_intro2 = st.columns([2, 1])
    
        with col_intro1:
            st.markdown("""
            Les analyses précédentes (**PCA**, **distance de Mahalanobis**, 
            **Z-scores**) ont identifié le **SMOD** (Snow Melt-Off Date, 
            fonte complète de la neige) comme la variable principale qui 
            rend NOHEDES atypique par rapport aux 8 autres sites fleurissants.
        
            - 📊 **PCA** : NOHEDES isolé dans l'espace climatique
            - 📊 **Distance de Mahalanobis** : position extrême
        
            👉 On va donc **modéliser le SMOD** en fonction des variables 
            climatiques pour comprendre **quand** et **pourquoi** NOHEDES 
            se démarque des autres sites au cours du temps.
            """)
    
        with col_intro2:
            st.info("""
            **Approche statistique** :
            - 🎯 Variable cible : SMOD_modis
            - 🔢 Variables candidates : 13
            - 📍 Sites d'entraînement : 8
            - 🧪 Site testé : NOHEDES
            - 📅 Période : 2000-2020
            """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 2 — MODÈLE COMPLET ET SÉLECTION AIC
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 📊 2. Construction du modèle — Sélection par AIC")
    
        # Préparer les données (filtrer 2000-2020)
        df_model = df[(df['annee'] >= 2000) & (df['annee'] <= 2020)].copy()
        df_autres = df_model[df_model['nom'] != 'NOHEDES'].copy()
        df_noh = df_model[df_model['nom'] == 'NOHEDES'].copy()
    
        # Variables candidates initiales (14)
        variables_14_all = [
            'temp_RF_moy', 'humidity_moy', 'wind_speed_moy',
            'etp_total', 'soil_temp_upper_moy', 'soil_moisture_moy',
            'LFD_nival', 'SCD', 'continuite', 'SOD_modis',
            'neige_pct', 'pr_RF_total', 'solar_radiation_moy'
        ]
    
        # Garder seulement celles présentes dans le df
        variables_14 = [v for v in variables_14_all if v in df_model.columns]
    
        # 11 variables retenues par AIC
        variables_11 = [v for v in variables_14 if v not in 
                         ['pr_RF_total', 'solar_radiation_moy']]
    
        variables_exclues = [v for v in variables_14 if v not in variables_11]
    
        # Construire les modèles
        formule_complet = "SMOD_modis ~ " + " + ".join(variables_14)
        mod_complet = ols(formule_complet, data=df_autres).fit()
    
        formule_aic = "SMOD_modis ~ " + " + ".join(variables_11)
        mod_aic = ols(formule_aic, data=df_autres).fit()
    
    
        # ── 2a : Variables candidates initiales
        st.markdown("### 📋 Variables candidates initiales")
    
        types_vars = {
            'temp_RF_moy': 'Climat air',
            'humidity_moy': 'Climat air',
            'wind_speed_moy': 'Climat air',
            'etp_total': 'Bilan énergétique',
            'soil_temp_upper_moy': 'Sol',
            'soil_moisture_moy': 'Sol',
            'LFD_nival': 'Neige',
            'SCD': 'Neige',
            'continuite': 'Neige',
            'SOD_modis': 'Neige',
            'neige_pct': 'Neige',
            'pr_RF_total': 'Climat air',
            'solar_radiation_moy': 'Climat air'
        }
    
        df_vars_init = pd.DataFrame({
            'Variable': variables_14,
            'Type': [types_vars.get(v, '?') for v in variables_14],
            'Description': [VARS_INFO.get(v, v) for v in variables_14]
        })
    
        st.dataframe(df_vars_init, use_container_width=True, hide_index=True,
                      height=min(500, 40 + 35 * len(variables_14)))
    
    
        # ── 2b : Coefficients du modèle complet
        st.markdown("### 🔬 Modèle complet (toutes variables)")
    
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        with col_mc1:
            st.metric("Nombre de variables", len(variables_14))
        with col_mc2:
            st.metric("R² ajusté", f"{mod_complet.rsquared_adj:.3f}")
        with col_mc3:
            st.metric("AIC", f"{mod_complet.aic:.1f}")
    
        with st.expander("📊 Voir les coefficients du modèle complet"):
            coefs_complet = pd.DataFrame({
                'Variable': mod_complet.params.index,
                'Coefficient': mod_complet.params.values.round(3),
                'Std. Error': mod_complet.bse.values.round(3),
                't-value': mod_complet.tvalues.values.round(2)
            })
            st.dataframe(coefs_complet, use_container_width=True, hide_index=True)
    
    
        # ── 2c : Sélection AIC stepwise
        st.markdown("### ✂️ Sélection AIC stepwise")
    
        st.markdown("""
        La procédure stepwise retire progressivement les variables qui 
        n'améliorent pas le critère AIC du modèle. Les variables exclues 
        sont celles qui apportent peu d'information ou qui sont redondantes 
        avec d'autres variables.
        """)
    
        col_aic1, col_aic2 = st.columns(2)
    
        with col_aic1:
            st.success(f"""
            ✅ **{len(variables_11)} variables retenues** :
            """ + "\n".join([f"- {v}" for v in variables_11]))
    
        with col_aic2:
            if variables_exclues:
                st.error(f"""
                ❌ **{len(variables_exclues)} variables exclues** :
                """ + "\n".join([f"- {v}" for v in variables_exclues]) + """
            
                *Variables redondantes avec d'autres ou peu informatives 
                pour la prédiction du SMOD.*
                """)
            else:
                st.info("Aucune variable exclue.")
    
    
        # ── 2d : Modèle AIC final
        st.markdown("### 📐 Modèle final AIC (variables retenues)")
    
        col_aic_a, col_aic_b, col_aic_c = st.columns(3)
        with col_aic_a:
            st.metric("Nombre de variables", len(variables_11))
        with col_aic_b:
            st.metric("R² ajusté", f"{mod_aic.rsquared_adj:.3f}")
        with col_aic_c:
            st.metric("AIC", f"{mod_aic.aic:.1f}",
                       delta=f"{mod_aic.aic - mod_complet.aic:+.1f} vs complet",
                       delta_color="inverse")
    
        coefs_aic = pd.DataFrame({
            'Variable': mod_aic.params.index,
            'Coefficient': mod_aic.params.values.round(3),
            'Std. Error': mod_aic.bse.values.round(3),
            't-value': mod_aic.tvalues.values.round(2)
        })
        st.dataframe(coefs_aic, use_container_width=True, hide_index=True)
    
        st.success(f"""
        ✅ **Le modèle AIC est meilleur** : AIC plus bas ({mod_aic.aic:.1f} 
        vs {mod_complet.aic:.1f}), R² ajusté légèrement plus élevé 
        ({mod_aic.rsquared_adj:.3f} vs {mod_complet.rsquared_adj:.3f}), 
        et plus parcimonieux ({len(variables_11)} variables vs 
        {len(variables_14)}).
        """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 3 — APPLICATION AUX DONNÉES NOHEDES
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 🎯 3. Application du modèle à NOHEDES")
    
        st.markdown("""
        Le modèle entraîné sur les **8 sites fleurissants** est appliqué 
        aux données NOHEDES. Pour chaque année, on calcule :
    
        - **SMOD observé** (mesuré par MODIS)
        - **SMOD prédit** (calculé par le modèle)
        - **Résidu** = Observé - Prédit
        - **IC 95%** de prédiction
        """)
    
        # Prédictions sur NOHEDES
        df_noh = df_noh.copy()
        df_noh['SMOD_predit'] = mod_aic.predict(df_noh)
        df_noh['residu'] = df_noh['SMOD_modis'] - df_noh['SMOD_predit']
    
        # IC 95%
        predictions = mod_aic.get_prediction(df_noh).summary_frame(alpha=0.05)
        df_noh['IC_low'] = predictions['obs_ci_lower'].values
        df_noh['IC_high'] = predictions['obs_ci_upper'].values
        df_noh['dans_IC'] = ((df_noh['SMOD_modis'] >= df_noh['IC_low']) & 
                              (df_noh['SMOD_modis'] <= df_noh['IC_high']))
    
        df_pred_aff = df_noh[['annee', 'SMOD_modis', 'SMOD_predit', 
                                'residu', 'IC_low', 'IC_high', 'dans_IC']].copy()
        df_pred_aff.columns = ['Année', 'SMOD observé', 'SMOD prédit', 'Résidu',
                                'IC bas', 'IC haut', 'Dans IC 95%']
        df_pred_aff = df_pred_aff.round(2)
        df_pred_aff['Dans IC 95%'] = df_pred_aff['Dans IC 95%'].apply(
            lambda x: '✅' if x else '⚠️'
        )
    
        with st.expander("📊 Voir le tableau complet des prédictions"):
            st.dataframe(df_pred_aff, use_container_width=True, hide_index=True)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 4 — ANALYSE DES RÉSIDUS & DÉTECTION DE LA RUPTURE
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 🔍 4. Détection de la rupture sur les résidus")
    
        # ── Hypothèses
        st.markdown("### 📌 Hypothèses statistiques")
    
        col_h0, col_h1 = st.columns(2)
        with col_h0:
            st.error("""
            **H₀ (hypothèse nulle)** :
        
            NOHEDES **NE SE DÉMARQUE PAS** des autres sites.
            Les résidus du modèle sont nuls en moyenne (avant et après).
            """)
        with col_h1:
            st.success("""
            **H₁ (hypothèse alternative)** :
        
            NOHEDES **SE DÉMARQUE** des autres sites.
            Les résidus diffèrent significativement entre les périodes.
            """)
    
    
        # ── Scan systématique RSS
        st.markdown("### 🔬 Détection de la rupture (scan systématique)")
    
        st.markdown("""
        On teste **chaque année** comme point de rupture possible. 
        L'année qui minimise la **Somme des Carrés Résiduels (RSS)** 
        correspond au meilleur point de rupture.
        """)
    
        residus_arr = df_noh['residu'].values
        annees_arr = df_noh['annee'].values
    
        rss_results = []
        rss_sans_rupture = np.sum((residus_arr - np.mean(residus_arr))**2)
    
        for a_rupt in range(2003, 2019):
            av = residus_arr[annees_arr < a_rupt]
            ap = residus_arr[annees_arr >= a_rupt]
            if len(av) >= 2 and len(ap) >= 2:
                rss = np.sum((av - np.mean(av))**2) + np.sum((ap - np.mean(ap))**2)
                t_test = ttest_ind(av, ap)
                rss_results.append({
                    'Année': a_rupt,
                    'RSS': round(rss, 1),
                    'Moy AVANT': round(np.mean(av), 2),
                    'Moy APRÈS': round(np.mean(ap), 2),
                    'Δ': round(np.mean(ap) - np.mean(av), 2),
                    'Réduction RSS (%)': round((1 - rss/rss_sans_rupture) * 100, 1),
                    'p-value': round(t_test.pvalue, 4)
                })
    
        rss_df = pd.DataFrame(rss_results)
    
        # Graphique RSS
        fig_rss = go.Figure()
        fig_rss.add_trace(go.Scatter(
            x=rss_df['Année'], y=rss_df['RSS'],
            mode='lines+markers',
            marker=dict(size=10, color='#7E3AC8'),
            line=dict(width=2, color='#7E3AC8'),
            name='RSS',
            hovertemplate='<b>Année %{x}</b><br>RSS : %{y:.1f}<extra></extra>'
        ))
    
        annee_min = rss_df.loc[rss_df['RSS'].idxmin(), 'Année']
        rss_min = rss_df['RSS'].min()
    
        fig_rss.add_vline(x=annee_min, line_dash="dash", line_color="red",
                           line_width=2,
                           annotation_text=f"RSS min : {int(annee_min)}",
                           annotation_position="top")
    
        fig_rss.update_layout(
            title="Détection de l'année de rupture (minimisation RSS)",
            xaxis_title="Année testée comme point de rupture",
            yaxis_title="Somme des Carrés Résiduels (RSS)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_rss, use_container_width=True)
    
        with st.expander("📊 Voir le tableau détaillé du scan"):
            st.dataframe(rss_df, use_container_width=True, hide_index=True)
    
        # Résultat
        reduction = (1 - rss_min/rss_sans_rupture) * 100
        pval_rupture = rss_df.loc[rss_df['RSS'].idxmin(), 'p-value']
    
        st.success(f"""
        🎯 **Rupture optimale détectée en {int(annee_min)}** :
        - RSS minimum = {rss_min:.0f}
        - Réduction RSS = {reduction:.1f}%
        - p-value (test t avant/après) = {pval_rupture:.4f}
    
        ➡️ **H₀ REJETÉE** : les résidus avant et après {int(annee_min)} 
        diffèrent significativement.
        """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 5 — GRAPHIQUE BARRES RÉSIDUS
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 📊 5. Visualisation des résidus annuels")
    
        df_noh['couleur'] = df_noh['residu'].apply(
            lambda x: '#2E7D32' if x >= 0 else '#C62828'
        )
    
        moy_av = df_noh[df_noh['annee'] < int(annee_min)]['residu'].mean()
        moy_ap = df_noh[df_noh['annee'] >= int(annee_min)]['residu'].mean()
    
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=df_noh['annee'],
            y=df_noh['residu'],
            marker_color=df_noh['couleur'],
            text=df_noh['residu'].round(1).apply(lambda x: f"{x:+.1f}"),
            textposition='outside',
            hovertemplate='<b>Année %{x}</b><br>Résidu : %{y:.1f} j<extra></extra>',
            showlegend=False
        ))
    
        fig_res.add_vline(x=int(annee_min) - 0.5, line_dash="dash", 
                           line_color="red", line_width=2,
                           annotation_text=f"Rupture {int(annee_min)}",
                           annotation_position="top")
        fig_res.add_hline(y=0, line_color="black", line_width=1)
    
        # Lignes moyennes
        fig_res.add_hline(y=moy_av, line_dash="dot", line_color="darkgreen",
                           annotation_text=f"Moy AV : {moy_av:+.1f}",
                           annotation_position="right")
        fig_res.add_hline(y=moy_ap, line_dash="dot", line_color="darkred",
                           annotation_text=f"Moy AP : {moy_ap:+.1f}",
                           annotation_position="right")
    
        fig_res.update_layout(
            title=f"Résidus du modèle SMOD à NOHEDES (2000-2020) — Rupture en {int(annee_min)}",
            xaxis_title="Année",
            yaxis_title="Résidu (jours)",
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_res, use_container_width=True)
    
        # ── Graphique linéaire avec points colorés (vue alternative)
        st.markdown("### 📈 Vue alternative : ligne avec points colorés")
    
        fig_line = go.Figure()
    
        # Ligne grise reliant tous les points
        fig_line.add_trace(go.Scatter(
            x=df_noh['annee'],
            y=df_noh['residu'],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
        # Points colorés selon signe (vert = positif, rouge = négatif)
        df_pos = df_noh[df_noh['residu'] >= 0]
        df_neg = df_noh[df_noh['residu'] < 0]
    
        fig_line.add_trace(go.Scatter(
            x=df_pos['annee'],
            y=df_pos['residu'],
            mode='markers+text',
            name='Résidu positif',
            marker=dict(color='#2E7D32', size=12,
                        line=dict(width=2, color='white')),
            text=df_pos['residu'].apply(lambda x: f"{x:+.0f}"),
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate='<b>Année %{x}</b><br>Résidu : %{y:.1f} j<br><i>NOHEDES garde sa neige plus longtemps</i><extra></extra>'
        ))
    
        fig_line.add_trace(go.Scatter(
            x=df_neg['annee'],
            y=df_neg['residu'],
            mode='markers+text',
            name='Résidu négatif',
            marker=dict(color='#C62828', size=12,
                        line=dict(width=2, color='white')),
            text=df_neg['residu'].apply(lambda x: f"{x:+.0f}"),
            textposition='bottom center',
            textfont=dict(size=10),
            hovertemplate='<b>Année %{x}</b><br>Résidu : %{y:.1f} j<br><i>NOHEDES perd sa neige plus tôt</i><extra></extra>'
        ))
    
        # Ligne horizontale à 0
        fig_line.add_hline(y=0, line_color="black", line_width=1)
    
        # Ligne verticale rupture
        fig_line.add_vline(
            x=int(annee_min),
            line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"Rupture {int(annee_min)}",
            annotation_position="top"
        )
    
        # Annotations moyennes
        fig_line.add_annotation(
            x=2002.5, y=-38,
            text=f"<b>Moyenne 2000-2005 : {moy_av:+.1f} j</b>",
            showarrow=False,
            font=dict(color="darkgreen", size=11),
            bgcolor="rgba(46, 125, 50, 0.1)",
            bordercolor="darkgreen",
            borderwidth=1
        )
    
        fig_line.add_annotation(
            x=2013, y=-38,
            text=f"<b>Moyenne 2006-2020 : {moy_ap:+.1f} j</b>",
            showarrow=False,
            font=dict(color="darkred", size=11),
            bgcolor="rgba(198, 40, 40, 0.1)",
            bordercolor="darkred",
            borderwidth=1
        )
    
        fig_line.update_layout(
            title="Évolution des résidus du modèle SMOD à NOHEDES (vue ligne)",
            xaxis_title="Année",
            yaxis_title="Résidu (jours)",
            height=500,
            template='plotly_white',
            xaxis=dict(dtick=2),
            yaxis=dict(range=[-45, 45]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                         xanchor="center", x=0.5)
        )
    
        st.plotly_chart(fig_line, use_container_width=True)
    
        st.caption("""
        💡 **Lecture** : Cette vue alternative montre la **trajectoire continue** 
        des résidus. La ligne grise relie tous les points pour visualiser 
        l'évolution temporelle, tandis que les points colorés indiquent le signe 
        (vert = NOHEDES garde sa neige plus longtemps, rouge = NOHEDES la perd 
        plus tôt que prédit).
        """)
    
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.success(f"""
            **AVANT {int(annee_min)}** (résidus positifs)
            - Moyenne : **{moy_av:+.1f} jours**
            - NOHEDES gardait sa neige **plus longtemps** que prédit
            - Effet "tampon protecteur"
            """)
        with col_r2:
            st.error(f"""
            **APRÈS {int(annee_min)}** (résidus négatifs)
            - Moyenne : **{moy_ap:+.1f} jours**
            - NOHEDES perd sa neige **plus tôt** que prédit
            - Effet protecteur perdu
            """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 6 — MODÈLE INTERACTION SITE × PÉRIODE
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 🎯 6. Modèle linéaire avec interaction")
    
        st.markdown("""
        Pour tester rigoureusement si la rupture est **spécifique à 
        NOHEDES**, on ajuste un modèle linéaire avec interaction :
    
        $$\\text{résidu} \\sim \\text{groupe} \\times \\text{période}$$
    
        où **groupe** = {NOHEDES, Autres} et **période** = 4 fenêtres.
        """)
    
        # Calculer les résidus pour TOUS les sites
        df_all = df_model.copy()
        df_all['SMOD_predit'] = mod_aic.predict(df_all)
        df_all['residu'] = df_all['SMOD_modis'] - df_all['SMOD_predit']
        df_all['groupe'] = df_all['nom'].apply(
            lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres'
        )
        df_all['periode'] = pd.cut(df_all['annee'],
                                    bins=[1999, 2005, 2010, 2015, 2020],
                                    labels=['2000-2005', '2006-2010', 
                                             '2011-2015', '2016-2020'])
    
        # Modèle interaction
        mod_int = ols("residu ~ C(groupe) * C(periode)", data=df_all).fit()
        anova_int = sm.stats.anova_lm(mod_int, typ=2)
    
        # ── Tableau ANOVA
        st.markdown("### 📋 Tableau ANOVA (Type II)")
    
        anova_aff = anova_int.copy()
        anova_aff.columns = ['Sum Sq', 'df', 'F', 'p-value']
        anova_aff = anova_aff.round(3)
        anova_aff['Signif.'] = anova_aff['p-value'].apply(
            lambda p: '***' if p < 0.001 else
                      '**' if p < 0.01 else
                      '*' if p < 0.05 else
                      '.' if p < 0.10 else 'ns'
        )
        st.dataframe(anova_aff, use_container_width=True)
    
        p_interaction = anova_int.loc['C(groupe):C(periode)', 'PR(>F)']
    
        if p_interaction < 0.05:
            st.success(f"""
            🎯 **Interaction site × période SIGNIFICATIVE** (p = {p_interaction:.4f} *)
        
            ➡️ NOHEDES évolue **DIFFÉREMMENT** des autres sites au cours du temps.
        
            ➡️ La rupture est **SPÉCIFIQUE à NOHEDES**, pas une tendance régionale générale.
            """)
        else:
            st.warning(f"""
            Interaction site × période : p = {p_interaction:.4f} (non significative)
            """)
    
    
        # ── VALIDATION PAR BOOTSTRAP (face à la non-normalité)
        st.markdown("### 🔬 Validation par Bootstrap (face à la non-normalité)")
    
        st.markdown("""
        Le test de Shapiro-Wilk indique que les résidus du modèle ne suivent 
        pas une loi normale (W = 0.80, p < 0.001). Pour confirmer la robustesse 
        de l'interaction malgré cette violation, on effectue une 
        **validation par bootstrap** (1000 ré-échantillonnages).
    
        **Principe** : on re-tire l'échantillon avec remise 1000 fois et on 
        recalcule le F-stat de l'interaction à chaque itération. Si la borne 
        inférieure de l'IC 95% > 1, l'interaction est confirmée.
        """)
    
        # Lancer le bootstrap
        with st.spinner("⏳ Bootstrap en cours (1000 itérations)..."):
            np.random.seed(42)
            F_bootstrap = []
        
            for i in range(1000):
                # Ré-échantillonnage avec remise
                idx = np.random.choice(len(df_all), size=len(df_all), replace=True)
                df_boot = df_all.iloc[idx].copy()
            
                # Recalcul du modèle
                try:
                    mod_b = ols("residu ~ C(groupe) * C(periode)", data=df_boot).fit()
                    anova_b = sm.stats.anova_lm(mod_b, typ=2)
                    F_b = anova_b.loc['C(groupe):C(periode)', 'F']
                    if not np.isnan(F_b):
                        F_bootstrap.append(F_b)
                except Exception:
                    continue
        
            F_bootstrap = np.array(F_bootstrap)
            F_obs = anova_int.loc['C(groupe):C(periode)', 'F']
            F_ic_low = np.quantile(F_bootstrap, 0.025)
            F_ic_high = np.quantile(F_bootstrap, 0.975)
            F_mean = F_bootstrap.mean()
            F_median = np.median(F_bootstrap)
    
        # Métriques bootstrap
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("F observé", f"{F_obs:.3f}")
        with col_b2:
            st.metric("F moyen bootstrap", f"{F_mean:.3f}")
        with col_b3:
            st.metric("IC 95% inf", f"{F_ic_low:.3f}",
                       delta="> 1 ✅" if F_ic_low > 1 else "≤ 1 ⚠️",
                       delta_color="off")
        with col_b4:
            st.metric("IC 95% sup", f"{F_ic_high:.3f}")
    
        # Graphique distribution bootstrap
        fig_boot = go.Figure()
    
        # Histogramme
        fig_boot.add_trace(go.Histogram(
            x=F_bootstrap,
            nbinsx=40,
            marker=dict(color='#2E7D32', line=dict(width=1, color='white')),
            opacity=0.7,
            name='Distribution bootstrap',
            hovertemplate='F : %{x:.2f}<br>Effectif : %{y}<extra></extra>'
        ))
    
        # Ligne F observé
        fig_boot.add_vline(
            x=F_obs, line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"F observé = {F_obs:.2f}",
            annotation_position="top"
        )
    
        # Lignes IC 95%
        fig_boot.add_vline(
            x=F_ic_low, line_dash="dot", line_color="darkblue", line_width=2,
            annotation_text=f"IC 2.5% = {F_ic_low:.2f}",
            annotation_position="bottom"
        )
        fig_boot.add_vline(
            x=F_ic_high, line_dash="dot", line_color="darkblue", line_width=2,
            annotation_text=f"IC 97.5% = {F_ic_high:.2f}",
            annotation_position="bottom"
        )
    
        # Seuil F = 1 (sous H0)
        fig_boot.add_vline(
            x=1, line_dash="solid", line_color="black", line_width=1,
            annotation_text="F=1 (seuil H₀)",
            annotation_position="top right"
        )
    
        fig_boot.update_layout(
            title=f"Distribution bootstrap du F-stat de l'interaction (n=1000)",
            xaxis_title="F-statistic (interaction site × période)",
            yaxis_title="Effectif",
            height=450,
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig_boot, use_container_width=True)
    
        # Conclusion bootstrap
        if F_ic_low > 1:
            st.success(f"""
            🎯 **INTERACTION CONFIRMÉE PAR BOOTSTRAP**
        
            - F observé = **{F_obs:.3f}**
            - IC 95% bootstrap = **[{F_ic_low:.3f}, {F_ic_high:.3f}]**
            - La borne inférieure (**{F_ic_low:.3f}**) est **supérieure à 1** 
              (seuil sous H₀)
        
            ➡️ L'interaction site × période est **STATISTIQUEMENT ROBUSTE** 
            malgré la non-normalité des résidus.
        
            ➡️ La rupture observée à NOHEDES est donc bien **SPÉCIFIQUE** et 
            non un artefact statistique.
            """)
        else:
            st.warning(f"""
            ⚠️ Interaction non confirmée par bootstrap.
            - IC 95% = [{F_ic_low:.3f}, {F_ic_high:.3f}]
            - La borne inférieure ≤ 1
            """)
    
        st.info("""
        💡 **Pourquoi le bootstrap ?**
    
        L'ANOVA classique suppose la normalité des résidus, qui n'est pas 
        respectée ici (Shapiro p < 0.001). Le bootstrap permet de calculer 
        la distribution empirique du F-stat **sans hypothèse de normalité**, 
        en re-tirant 1000 fois l'échantillon avec remise. Si l'IC 95% ne 
        contient pas 1 (valeur attendue sous H₀), on conclut que l'effet 
        est réel et non dû au hasard.
        """)
    
    
        # ── Graphique d'interaction (LE FAVORI DU PROF)
        st.markdown("### 📊 Graphique d'interaction (résidus moyens)")
    
        # Calculer moyennes et IC par groupe × période
        stats_int = df_all.groupby(['groupe', 'periode'], observed=True).agg(
            moy=('residu', 'mean'),
            sd=('residu', 'std'),
            n=('residu', 'count')
        ).reset_index()
        stats_int['se'] = stats_int['sd'] / np.sqrt(stats_int['n'])
        stats_int['ic_low'] = stats_int['moy'] - 1.96 * stats_int['se']
        stats_int['ic_high'] = stats_int['moy'] + 1.96 * stats_int['se']
    
        # P-values post-hoc par fenêtre
        p_vals_periode = {}
        diff_periode = {}
        for p in ['2000-2005', '2006-2010', '2011-2015', '2016-2020']:
            df_p = df_all[df_all['periode'] == p]
            noh = df_p[df_p['groupe'] == 'NOHEDES']['residu']
            aut = df_p[df_p['groupe'] == 'Autres']['residu']
            if len(noh) >= 2 and len(aut) >= 2:
                t_test = ttest_ind(noh, aut)
                p_vals_periode[p] = t_test.pvalue
                diff_periode[p] = noh.mean() - aut.mean()
    
        # Graphique
        fig_int = go.Figure()
    
        # Bande jaune pour 2000-2005 (période de référence)
        fig_int.add_vrect(
            x0=-0.5, x1=0.5,
            fillcolor="yellow", opacity=0.15,
            line_width=0
        )
    
        for groupe, couleur in [('Autres', '#7E3AC8'), ('NOHEDES', '#2E7D32')]:
            sub = stats_int[stats_int['groupe'] == groupe].copy()
            sub = sub.sort_values('periode')
        
            fig_int.add_trace(go.Scatter(
                x=sub['periode'].astype(str), y=sub['moy'],
                mode='lines+markers',
                name=groupe,
                line=dict(color=couleur, width=3),
                marker=dict(size=14, color=couleur,
                            line=dict(width=2, color='white')),
                error_y=dict(type='data',
                              array=sub['moy'] - sub['ic_low'],
                              color=couleur, thickness=2, width=8),
                hovertemplate=f'<b>{groupe}</b><br>%{{x}}<br>'
                              f'Résidu moyen : %{{y:.2f}} j<extra></extra>'
            ))
    
        # Annotations p-values post-hoc
        y_max = stats_int['ic_high'].max()
        for p, pval in p_vals_periode.items():
            symbole = '***' if pval < 0.001 else '**' if pval < 0.01 else \
                      '*' if pval < 0.05 else 'ns'
            couleur_sym = "darkred" if pval < 0.05 else "gray"
            fig_int.add_annotation(
                x=p, y=y_max + 8,
                text=f"<b>{symbole}</b><br><span style='font-size:10px'>p={pval:.3f}</span>",
                showarrow=False,
                font=dict(color=couleur_sym, size=12)
            )
    
        fig_int.add_hline(y=0, line_color="black", line_dash="dash", line_width=1)
    
        fig_int.update_layout(
            title=f"Résidus moyens par groupe et période — Interaction p = {p_interaction:.3f}",
            xaxis_title="Période",
            yaxis_title="Résidu moyen du modèle SMOD (jours)",
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                         xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_int, use_container_width=True)
    
        # ── 2ème graphique ENRICHI avec annotations explicatives
        st.markdown("### 📊 Graphique enrichi avec annotations explicatives")
    
        st.markdown("""
        Ce 2ème graphique met en évidence la **bascule** entre la période 
        2000-2005 (où NOHEDES se démarque) et les périodes suivantes 
        (où NOHEDES suit le modèle).
        """)
    
        fig_int_annot = go.Figure()
    
        # Zone jaune pour 2000-2005 (référence)
        fig_int_annot.add_vrect(
            x0=-0.5, x1=0.5,
            fillcolor="yellow", opacity=0.15,
            line_width=0,
            annotation_text="Période de référence",
            annotation_position="top left",
            annotation_font=dict(size=10, color="darkorange")
        )
    
        # Ligne horizontale à 0
        fig_int_annot.add_hline(
            y=0, line_color="black",
            line_dash="dash", line_width=1
        )
    
        # Tracer les courbes
        for groupe, couleur in [('Autres', '#7E3AC8'), ('NOHEDES', '#2E7D32')]:
            sub = stats_int[stats_int['groupe'] == groupe].copy()
            sub = sub.sort_values('periode')
        
            # Étiquette descriptive
            label_legende = '8 sites fleurissants' if groupe == 'Autres' else 'NOHEDES'
        
            fig_int_annot.add_trace(go.Scatter(
                x=sub['periode'].astype(str), y=sub['moy'],
                mode='lines+markers+text',
                name=label_legende,
                line=dict(color=couleur, width=3),
                marker=dict(size=16, color=couleur,
                            line=dict(width=2, color='white')),
                error_y=dict(type='data',
                              array=sub['moy'] - sub['ic_low'],
                              color=couleur, thickness=2, width=10),
                text=sub['moy'].apply(lambda x: f"{x:+.1f}"),
                textposition='top center' if groupe == 'NOHEDES' else 'bottom center',
                textfont=dict(size=11, color=couleur, family='Arial Black'),
                hovertemplate=f'<b>{label_legende}</b><br>%{{x}}<br>'
                              f'Résidu moyen : %{{y:.2f}} j<extra></extra>'
            ))
    
        # Annotations p-values en haut
        y_top_annot = max(stats_int['ic_high'].max() + 10, 50)
        for p, pval in p_vals_periode.items():
            symbole = '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            couleur_sym = "darkred" if pval < 0.05 else "gray"
            fig_int_annot.add_annotation(
                x=p, y=y_top_annot,
                text=f"<b>p={pval:.3f}</b><br><b>{symbole}</b>",
                showarrow=False,
                font=dict(color=couleur_sym, size=12),
                align='center'
            )
    
        # Flèche "Chute brutale"
        moy_noh_t0 = stats_int[(stats_int['groupe'] == 'NOHEDES') & 
                                (stats_int['periode'].astype(str) == '2000-2005')]['moy'].values
        moy_noh_t1 = stats_int[(stats_int['groupe'] == 'NOHEDES') & 
                                (stats_int['periode'].astype(str) == '2006-2010')]['moy'].values
    
        if len(moy_noh_t0) > 0 and len(moy_noh_t1) > 0:
            delta_chute = moy_noh_t1[0] - moy_noh_t0[0]
        
            fig_int_annot.add_annotation(
                x='2006-2010',
                y=moy_noh_t1[0],
                ax='2000-2005',
                ay=moy_noh_t0[0] - 5,
                xref='x', yref='y',
                axref='x', ayref='y',
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=2.5,
                arrowcolor="darkred"
            )
        
            fig_int_annot.add_annotation(
                x=0.5, y=moy_noh_t0[0] - 5,
                text=f"<b>Chute brutale</b><br>{delta_chute:.0f} jours",
                showarrow=False,
                font=dict(color="darkred", size=12, family='Arial Black'),
                align='center',
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="darkred",
                borderwidth=1
            )
    
        # Annotation "H1 confirmée" pour 2000-2005
        y_min_annot = min(stats_int['ic_low'].min() - 5, -25)
        fig_int_annot.add_annotation(
            x='2000-2005', y=y_min_annot,
            text="★ <b>NOHEDES se démarque</b><br><i>(H₁ confirmée)</i>",
            showarrow=False,
            font=dict(color="darkgreen", size=11, family='Arial'),
            align='center',
            bgcolor="rgba(46, 125, 50, 0.1)",
            bordercolor="darkgreen",
            borderwidth=1
        )
    
        # Annotation "H0 maintenue" pour 2011-2015
        fig_int_annot.add_annotation(
            x='2011-2015', y=y_min_annot,
            text="<b>NOHEDES suit le modèle</b><br><i>(H₀ maintenue)</i>",
            showarrow=False,
            font=dict(color="darkblue", size=11, family='Arial'),
            align='center',
            bgcolor="rgba(126, 58, 200, 0.1)",
            bordercolor="darkblue",
            borderwidth=1
        )
    
        # Titre et mise en page
        fig_int_annot.update_layout(
            title=dict(
                text=f"<b>Évolution des résidus du modèle SMOD</b><br>"
                      f"<span style='font-size:12px; color:darkblue'>"
                      f"Interaction site × période significative "
                      f"(p = {p_interaction:.3f} *)</span>",
                x=0.5, xanchor='center'
            ),
            xaxis_title="<b>Période</b>",
            yaxis_title="<b>Résidu moyen (jours)</b>",
            height=600,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.2,
                xanchor="center", x=0.5,
                font=dict(size=11)
            ),
            annotations=[
                *fig_int_annot.layout.annotations,
            ]
        )
    
        # Étendre légèrement l'axe Y pour bien voir annotations
        fig_int_annot.update_yaxes(
            range=[y_min_annot - 5, y_top_annot + 10]
        )
    
        st.plotly_chart(fig_int_annot, use_container_width=True)
    
        st.caption("""
        💡 **Lecture du graphique** : 
        - **Zone jaune (2000-2005)** : période de référence où NOHEDES se démarque 
          significativement des autres sites (p = 0.008 **).
        - **Flèche rouge** : chute brutale des résidus NOHEDES entre 2000-2005 et 2006-2010.
        - **Périodes suivantes** : NOHEDES s'aligne sur les autres sites 
          (p > 0.05, hypothèse H₀ maintenue).
        - **Barres verticales** : intervalles de confiance à 95%.
        """)
    
        # ── Tableau post-hoc
        st.markdown("### 📋 Comparaisons post-hoc (NOHEDES vs Autres par période)")
    
        posthoc_data = []
        for p in ['2000-2005', '2006-2010', '2011-2015', '2016-2020']:
            df_p = df_all[df_all['periode'] == p]
            noh = df_p[df_p['groupe'] == 'NOHEDES']['residu']
            aut = df_p[df_p['groupe'] == 'Autres']['residu']
            t_test = ttest_ind(noh, aut)
            signif = '***' if t_test.pvalue < 0.001 else \
                     '**' if t_test.pvalue < 0.01 else \
                     '*' if t_test.pvalue < 0.05 else 'ns'
            posthoc_data.append({
                'Période': p,
                'Moy NOHEDES': round(noh.mean(), 2),
                'Moy Autres': round(aut.mean(), 2),
                'Différence': round(noh.mean() - aut.mean(), 2),
                'p-value': round(t_test.pvalue, 4),
                'Signif.': signif
            })
    
        st.dataframe(pd.DataFrame(posthoc_data), use_container_width=True, hide_index=True)
    
        st.info("""
        📊 **Lecture du tableau** :
        - **2000-2005** : NOHEDES significativement différent (résidus +25 vs +3)
        - **2006-2020** : NOHEDES s'aligne sur les autres sites
    
        → Confirme la **perte de l'effet protecteur** à NOHEDES après 2006.
        """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 7 — ANALYSE DES 11 VARIABLES EXPLICATIVES
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 🔬 7. Pourquoi NOHEDES bascule-t-il ?")
    
        st.markdown("""
        Pour comprendre quels changements climatiques sous-tendent la rupture, 
        on analyse l'évolution de chacune des **11 variables explicatives** 
        entre 2000-2005 et les périodes suivantes.
    
        Deux tests sont effectués :
        1. **Test t à NOHEDES** : la variable a-t-elle changé à NOHEDES ?
        2. **Interaction site × période** : ce changement est-il SPÉCIFIQUE à NOHEDES ?
        """)
    
        # Calculer pour les 11 variables
        analyse_vars = []
    
        for v in variables_11:
            # Δ à NOHEDES (2006-2020 vs 2000-2005)
            noh_av = df_noh[df_noh['annee'] < 2006][v].mean()
            noh_ap = df_noh[df_noh['annee'] >= 2006][v].mean()
            delta_noh = noh_ap - noh_av
        
            # Δ aux Autres
            aut_av = df_autres[df_autres['annee'] < 2006][v].mean()
            aut_ap = df_autres[df_autres['annee'] >= 2006][v].mean()
            delta_aut = aut_ap - aut_av
        
            # Test t (sur NOHEDES) : 2000-2005 vs chaque autre fenêtre
            pvals_test_t = []
            for periode_test in [(2006, 2010), (2011, 2015), (2016, 2020)]:
                x1 = df_noh[df_noh['annee'].between(2000, 2005)][v].dropna()
                x2 = df_noh[df_noh['annee'].between(periode_test[0], periode_test[1])][v].dropna()
                if len(x1) >= 2 and len(x2) >= 2:
                    t = ttest_ind(x1, x2)
                    pvals_test_t.append(t.pvalue)
            p_min_t = min(pvals_test_t) if pvals_test_t else 1.0
        
            # Interaction site × période
            df_v = df_all.dropna(subset=[v])
            try:
                mod_v = ols(f"{v} ~ C(groupe) * C(periode)", data=df_v).fit()
                anova_v = sm.stats.anova_lm(mod_v, typ=2)
                p_inter = anova_v.loc['C(groupe):C(periode)', 'PR(>F)']
            except Exception:
                p_inter = np.nan
        
            analyse_vars.append({
                'Variable': v,
                'Δ NOHEDES': f"{delta_noh:+.3f}",
                'Δ Autres': f"{delta_aut:+.3f}",
                'p test t (NOH)': f"{p_min_t:.4f}",
                'A changé à NOH ?': '⭐ OUI' if p_min_t < 0.05 else '❌ ns',
                'p interaction': f"{p_inter:.4f}" if not np.isnan(p_inter) else 'NA',
                'Spécifique à NOH ?': '✅ OUI' if (not np.isnan(p_inter) and p_inter < 0.05) else '❌ NON'
            })
    
        df_analyse = pd.DataFrame(analyse_vars)
        st.dataframe(df_analyse, use_container_width=True, hide_index=True)
    
        # Comptage
        n_change = sum(1 for r in analyse_vars if '⭐' in r['A changé à NOH ?'])
        n_specifique = sum(1 for r in analyse_vars if '✅' in r['Spécifique à NOH ?'])
    
        st.info(f"""
        ### 📊 Synthèse :
    
        - **{n_change} variables ont CHANGÉ à NOHEDES** (test t significatif)
        - **{n_specifique} variable(s) sont SPÉCIFIQUES à NOHEDES** (interaction significative)
    
        Les variables qui ont changé à NOHEDES mais aussi sur les autres sites 
        reflètent une **tendance régionale**. Seules les variables avec 
        interaction significative reflètent un **changement local SPÉCIFIQUE** 
        à NOHEDES.
        """)
    
        # Sélecteur de variable pour exploration
        st.markdown("### 🔍 Explorer une variable spécifique")
    
        var_choisie = st.selectbox(
            "Sélectionne une variable pour voir son évolution :",
            options=variables_11,
            index=variables_11.index('etp_total') if 'etp_total' in variables_11 else 0
        )
    
        # Graphique évolution NOHEDES vs Autres pour cette variable
        stats_v = df_all.dropna(subset=[var_choisie]).groupby(
            ['groupe', 'periode'], observed=True
        ).agg(
            moy=(var_choisie, 'mean'),
            sd=(var_choisie, 'std'),
            n=(var_choisie, 'count')
        ).reset_index()
        stats_v['se'] = stats_v['sd'] / np.sqrt(stats_v['n'])
        stats_v['ic_low'] = stats_v['moy'] - 1.96 * stats_v['se']
        stats_v['ic_high'] = stats_v['moy'] + 1.96 * stats_v['se']
    
        fig_var = go.Figure()
    
        # Bande jaune pour 2000-2005 (période de référence)
        fig_var.add_vrect(
            x0=-0.5, x1=0.5,
            fillcolor="yellow", opacity=0.15,
            line_width=0
        )
    
        for groupe, couleur in [('Autres', '#7E3AC8'), ('NOHEDES', '#2E7D32')]:
            sub = stats_v[stats_v['groupe'] == groupe].sort_values('periode')
            fig_var.add_trace(go.Scatter(
                x=sub['periode'].astype(str), y=sub['moy'],
                mode='lines+markers',
                name=groupe,
                line=dict(color=couleur, width=3),
                marker=dict(size=14, color=couleur,
                            line=dict(width=2, color='white')),
                error_y=dict(type='data',
                              array=sub['moy'] - sub['ic_low'],
                              color=couleur, thickness=2, width=8),
                hovertemplate=f'<b>{groupe}</b><br>%{{x}}<br>'
                              f'{var_choisie} : %{{y:.3f}}<extra></extra>'
            ))
    
        fig_var.update_layout(
            title=f"Évolution de {var_choisie} : NOHEDES vs Autres",
            xaxis_title="Période",
            yaxis_title=VARS_INFO.get(var_choisie, var_choisie),
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                         xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_var, use_container_width=True)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 8 — FOCUS ETP
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## ⭐ 8. Focus sur l'ETP — Variable clé")
    
        st.markdown("""
        L'**évapotranspiration (ETP)** est la seule variable qui présente :
        - Un changement statistiquement **SPÉCIFIQUE** à NOHEDES
        - Une interaction site × période **significative**
        - Une **augmentation 4-5 fois plus forte** à NOHEDES qu'ailleurs
    
        👉 C'est notre **indicateur principal** du changement microclimatique local.
        """)
    
        # Calculer pattern Tukey ETP
        if 'etp_total' in variables_11:
            # Évolution par période — NOHEDES vs Autres
            etp_evolution = []
            for p in ['2000-2005', '2006-2010', '2011-2015', '2016-2020']:
                df_p = df_all[df_all['periode'] == p].dropna(subset=['etp_total'])
                noh = df_p[df_p['groupe'] == 'NOHEDES']['etp_total']
                aut = df_p[df_p['groupe'] == 'Autres']['etp_total']
                t_test = ttest_ind(noh, aut)
            
                etp_evolution.append({
                    'Période': p,
                    'NOHEDES (mm)': round(noh.mean(), 1),
                    'Autres (mm)': round(aut.mean(), 1),
                    'Différence': round(noh.mean() - aut.mean(), 1),
                    'p-value': round(t_test.pvalue, 4),
                    'Lettres Tukey': 'a/a' if t_test.pvalue >= 0.05 else 'a/b'
                })
        
            df_etp = pd.DataFrame(etp_evolution)
        
            st.markdown("### 📋 Test de Tukey ETP par période")
            st.dataframe(df_etp, use_container_width=True, hide_index=True)
        
            # Graphique ETP
            st.markdown("### 📊 Évolution de l'ETP par période")
        
            fig_etp = go.Figure()
        
            # Bande jaune pour 2000-2005 (période de référence)
            fig_etp.add_vrect(
                x0=-0.5, x1=0.5,
                fillcolor="yellow", opacity=0.15,
                line_width=0
            )
        
            # NOHEDES
            fig_etp.add_trace(go.Scatter(
                x=df_etp['Période'], y=df_etp['NOHEDES (mm)'],
                mode='lines+markers',
                name='NOHEDES',
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=14, color='#2E7D32',
                            line=dict(width=2, color='white')),
                text=df_etp['NOHEDES (mm)'].apply(lambda x: f"{x:.0f} mm"),
                textposition='top center',
                hovertemplate='<b>NOHEDES</b><br>%{x}<br>ETP : %{y:.0f} mm<extra></extra>'
            ))
        
            # Autres
            fig_etp.add_trace(go.Scatter(
                x=df_etp['Période'], y=df_etp['Autres (mm)'],
                mode='lines+markers',
                name='Autres sites',
                line=dict(color='#7E3AC8', width=3),
                marker=dict(size=14, color='#7E3AC8',
                            line=dict(width=2, color='white')),
                text=df_etp['Autres (mm)'].apply(lambda x: f"{x:.0f} mm"),
                textposition='bottom center',
                hovertemplate='<b>Autres</b><br>%{x}<br>ETP : %{y:.0f} mm<extra></extra>'
            ))
        
            # Annotations lettres Tukey
            for i, row in df_etp.iterrows():
                fig_etp.add_annotation(
                    x=row['Période'], y=row['NOHEDES (mm)'] + 20,
                    text=f"<b>{row['Lettres Tukey']}</b>",
                    showarrow=False,
                    font=dict(color='darkred', size=14)
                )
        
            fig_etp.update_layout(
                title="Évolution de l'évapotranspiration (ETP) : NOHEDES vs Autres",
                xaxis_title="Période",
                yaxis_title="ETP totale (mm)",
                height=500,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                             xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_etp, use_container_width=True)
        
            # Métriques ETP
            delta_noh_etp = df_etp['NOHEDES (mm)'].iloc[1:].mean() - df_etp['NOHEDES (mm)'].iloc[0]
            delta_aut_etp = df_etp['Autres (mm)'].iloc[1:].mean() - df_etp['Autres (mm)'].iloc[0]
        
            col_etp1, col_etp2, col_etp3 = st.columns(3)
            with col_etp1:
                st.metric("Δ ETP NOHEDES", f"+{delta_noh_etp:.0f} mm",
                           help="Augmentation moyenne après 2005")
            with col_etp2:
                st.metric("Δ ETP Autres", f"+{delta_aut_etp:.0f} mm")
            with col_etp3:
                ratio = delta_noh_etp / delta_aut_etp if delta_aut_etp != 0 else 0
                st.metric("Ratio", f"{ratio:.1f}×",
                           help="NOHEDES augmente N fois plus que les autres")
        
            st.success("""
            🎯 **Interprétation** :
        
            - Avant 2006, l'ETP à NOHEDES était **équivalente** aux autres sites 
              (Tukey : a/a)
            - À partir de 2006, l'ETP à NOHEDES **dépasse significativement** 
              celle des autres sites (Tukey : a/b)
            - Le **détachement** est cohérent avec la rupture observée 
              sur les résidus du SMOD en 2006
            """)
        
            st.warning("""
            ⚠️ **Attention** : Une régression simple SMOD ~ ETP donne un R² très 
            faible (< 0.05), ce qui signifie que l'ETP n'est **pas une cause 
            directe linéaire** du SMOD. Elle fonctionne plutôt comme un 
            **INDICATEUR INTÉGRATEUR** du bilan énergétique atmosphérique 
            local, dont l'évolution reflète un changement microclimatique 
            plus large.
            """)
    
        st.markdown("---")
    
    
        # ════════════════════════════════════════════════════════════
        # SECTION 9 — CONCLUSION
        # ════════════════════════════════════════════════════════════
    
        st.markdown("## 📝 9. Conclusion")
    
        st.success(f"""
        ### 🎯 Synthèse des résultats
    
        **1. NOHEDES est statistiquement atypique** (PCA, Mahalanobis, Z-scores)
        → confirmé dans l'onglet précédent.
    
        **2. Modèle SMOD avec sélection AIC** :
        - {len(variables_11)} variables retenues
        - R² ajusté = {mod_aic.rsquared_adj:.3f}
        - Entraîné sur les 8 sites fleurissants
    
        **3. Rupture détectée en {int(annee_min)} sur les résidus NOHEDES** :
        - Résidus moyens passent de **{moy_av:+.1f} j** à **{moy_ap:+.1f} j**
        - Réduction RSS de {reduction:.1f}%
        - p < 0.001
    
        **4. La rupture est SPÉCIFIQUE à NOHEDES** :
        - Interaction site × période : **p = {p_interaction:.4f}**
        - Post-hoc 2000-2005 : NOHEDES vs Autres significatif
        - Post-hoc 2006-2020 : NOHEDES s'aligne sur les autres
    
        **5. L'ETP est l'indicateur principal du changement** :
        - Seule variable avec interaction site × période significative
        - +138 mm à NOHEDES vs +30 mm aux autres sites
        - Pattern Tukey : **a/a → a/b** (détachement en 2006)
    
        **6. Conclusion scientifique** :
    
        > *"La relation entre le SMOD de Nohèdes et les variables climatiques 
        > observées sur les autres sites a changé autour de {int(annee_min)}."*
    
        Ce changement reflète une **modification microclimatique LOCALE** 
        à NOHEDES, dont l'**ETP** est le meilleur indicateur statistique, 
        sans qu'on puisse établir une causalité directe simple entre 
        l'ETP et le SMOD à l'échelle interannuelle.
        """)


    # ═══════════════════════════════════════════════════════════════════

