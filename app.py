"""
================================================================
APPLICATION STREAMLIT — Analyse NOHEDES
Version finale : PCA & Mahalanobis + Distance Euclidienne + SMOD + Évolution
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from scipy import stats
from scipy.stats import chi2

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="NOHEDES — Analyse",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .histoire-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #6A1B9A;
        margin: 15px 0;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFF8E1 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #E65100;
    }
    h1 { color: #1A237E; }
    h2 { color: #283593; margin-top: 30px; }
    h3 { color: #3949AB; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════

VARS_INFO = {
    "SMOD_modis": "Fin enneigement (jour nival)",
    "continuite": "Continuité du manteau neigeux",
    "temp_RF_moy": "Température air moyenne (°C)",
    "SCD": "Nombre de jours de neige",
    "humidity_moy": "Humidité air moyenne (%)",
    "neige_pct": "% précipitations",
    "soil_temp_upper_moy": "Température sol (°C)",
    "LFD_nival": "Dernier jour de neige (LFD)"
}

VARIABLES_X = list(VARS_INFO.keys())

FENETRES = {
    "F1 : 2000-2004": [2000, 2001, 2002, 2004],
    "F2 : 2005-2009": [2005, 2006, 2007, 2008, 2009],
    "F3 : 2010-2014": [2010, 2011, 2012, 2013, 2014],
    "F4 : 2015-2017": [2015, 2016, 2017],
    "F5 : 2018-2020": [2018, 2019, 2020]
}

COLOR_NOHEDES = "#2E7D32"
COLOR_AUTRES = "#7E3AC8"
COLOR_ELLIPSE = "rgba(126, 58, 200, 0.15)"
COLOR_ELLIPSE_LINE = "rgba(126, 58, 200, 0.7)"
COLOR_ARROW = "rgba(255, 140, 0, 0.85)"

DMAX_REF = 7.189

# ═══════════════════════════════════════════════════════
# FONCTIONS NMI (Niche Margin Index)
# ═══════════════════════════════════════════════════════

@st.cache_data
def calculer_niche_et_NMI():
    """
    Calcule la niche des 8 sites (KDE 2D dans PCA moyen)
    puis le NMI de NOHEDES pour chaque année.
    """
    from scipy.stats import gaussian_kde
    
    df_data = charger_donnees()
    
    # ─── ÉTAPE 1 : PCA moyen sur 21 ans ───
    df_moyen = df_data.groupby('nom')[VARIABLES_X].mean().reset_index()
    df_moyen = df_moyen.dropna()
    
    X_moy = df_moyen[VARIABLES_X].values
    scaler = StandardScaler()
    X_moy_std = scaler.fit_transform(X_moy)
    
    pca_ref = SklearnPCA(n_components=2)
    coords_ref = pca_ref.fit_transform(X_moy_std)
    
    idx_noh = list(df_moyen['nom']).index('NOHEDES')
    coord_noh_moy = coords_ref[idx_noh]
    coords_8sites = np.delete(coords_ref, idx_noh, axis=0)
    noms_8sites = df_moyen['nom'].drop(df_moyen.index[idx_noh]).values
    
    # ─── ÉTAPE 2 : Nuage des 8 sites × 21 ans ───
    df_all_8 = df_data[df_data['nom'] != 'NOHEDES'].copy()
    df_all_8 = df_all_8[['nom', 'annee'] + VARIABLES_X].dropna()
    
    X_all = df_all_8[VARIABLES_X].values
    X_all_std = scaler.transform(X_all)
    coords_all_8 = X_all_std @ pca_ref.components_[:2].T
    
    # ─── ÉTAPE 3 : KDE 2D ───
    kde = gaussian_kde(coords_all_8.T)
    densites_points = kde(coords_all_8.T)
    seuil_densite = np.quantile(densites_points, 0.05)  # 95% de la niche
    
    # ─── ÉTAPE 4 : Grille pour visualisation ───
    x_min, x_max = coords_all_8[:, 0].min() - 3, coords_all_8[:, 0].max() + 3
    y_min, y_max = coords_all_8[:, 1].min() - 3, coords_all_8[:, 1].max() + 3
    grid_x = np.linspace(x_min, x_max, 100)
    grid_y = np.linspace(y_min, y_max, 100)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([XX.ravel(), YY.ravel()])
    grid_densites = kde(grid_points).reshape(XX.shape)
    
    # ─── ÉTAPE 5 : Points de la marge (contour à seuil) ───
    # Utilise matplotlib (déjà installé, pas besoin de skimage)
    import matplotlib.pyplot as plt
    
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(XX, YY, grid_densites, levels=[seuil_densite])
    plt.close(fig_tmp)
    
    points_marge = []
    # Extraction des points du contour
    for collection in cs.allsegs[0] if hasattr(cs, 'allsegs') else []:
        for pt in collection:
            points_marge.append([pt[0], pt[1]])
    
    # Fallback pour compatibilité avec versions de matplotlib différentes
    if not points_marge:
        try:
            for path in cs.collections[0].get_paths():
                for vertex in path.vertices:
                    points_marge.append([vertex[0], vertex[1]])
        except Exception:
            pass
    
    points_marge = np.array(points_marge) if points_marge else np.array([[0, 0]])
    
    # ─── ÉTAPE 6 : dmax ───
    all_pts = np.vstack([coords_all_8, coord_noh_moy.reshape(1, -1)])
    from scipy.spatial.distance import pdist
    dmax_NMI = pdist(all_pts).max()
    
    # ─── ÉTAPE 7 : NMI par année pour NOHEDES ───
    resultats_NMI = []
    for an in sorted(df_data['annee'].unique()):
        noh_an = df_data[(df_data['nom'] == 'NOHEDES') & (df_data['annee'] == an)][VARIABLES_X].dropna()
        if len(noh_an) == 0:
            continue
        
        X_noh_an = noh_an.values
        X_noh_an_std = scaler.transform(X_noh_an)
        coord_noh_an = (X_noh_an_std @ pca_ref.components_[:2].T)[0]
        
        # Densité au point
        d_point = kde(coord_noh_an.reshape(-1, 1))[0]
        
        # Distance à la marge
        if len(points_marge) > 1:
            dists_marge = np.sqrt(np.sum((points_marge - coord_noh_an)**2, axis=1))
            d_marge = dists_marge.min()
        else:
            d_marge = 0
        
        # NMI
        if d_point >= seuil_densite:
            NMI = d_marge / dmax_NMI
            statut = "DANS niche"
        else:
            NMI = -d_marge / dmax_NMI
            statut = "HORS niche"
        
        resultats_NMI.append({
            'annee': an,
            'PC1_NOHEDES': coord_noh_an[0],
            'PC2_NOHEDES': coord_noh_an[1],
            'NMI': NMI,
            'statut': statut
        })
    
    df_NMI = pd.DataFrame(resultats_NMI)
    
    # Calculer NMI pour chaque case de la grille (pour le fond)
    grid_NMI = np.zeros_like(grid_densites)
    for i in range(grid_densites.shape[0]):
        for j in range(grid_densites.shape[1]):
            pt = np.array([XX[i, j], YY[i, j]])
            if len(points_marge) > 1:
                d_m = np.min(np.sqrt(np.sum((points_marge - pt)**2, axis=1)))
            else:
                d_m = 0
            if grid_densites[i, j] >= seuil_densite:
                grid_NMI[i, j] = d_m / dmax_NMI
            else:
                grid_NMI[i, j] = -d_m / dmax_NMI
    
    return {
        'coords_8sites': coords_8sites,
        'noms_8sites': noms_8sites,
        'coord_noh_moy': coord_noh_moy,
        'coords_all_8': coords_all_8,
        'XX': XX, 'YY': YY,
        'grid_densites': grid_densites,
        'grid_NMI': grid_NMI,
        'seuil_densite': seuil_densite,
        'points_marge': points_marge,
        'dmax_NMI': dmax_NMI,
        'df_NMI': df_NMI
    }



# ═══════════════════════════════════════════════════════
# FONCTIONS SMOD vs LFD
# ═══════════════════════════════════════════════════════

def jour_nival_to_date(jour_nival, annee_nivale=2020):
    """Convertit un jour nival en date calendaire (1er sept = jour 1)"""
    from datetime import datetime, timedelta
    
    if pd.isna(jour_nival):
        return ""
    
    debut = datetime(annee_nivale - 1, 9, 1)
    date = debut + timedelta(days=int(jour_nival) - 1)
    
    mois_fr = {
        1: "jan", 2: "fév", 3: "mars", 4: "avr", 5: "mai", 6: "juin",
        7: "juil", 8: "août", 9: "sept", 10: "oct", 11: "nov", 12: "déc"
    }
    return f"{date.day} {mois_fr[date.month]}"


def jour_nival_to_saison(jour_nival):
    """Détermine la saison à partir du jour nival (1er sept = jour 1)"""
    if pd.isna(jour_nival):
        return "?"
    
    j = int(jour_nival)
    if j <= 91:
        return "Automne (SON)"
    elif j <= 181:
        return "Hiver (DJF)"
    elif j <= 273:
        return "Printemps (MAM)"
    else:
        return "Été (JJA)"


def plot_evolution_smod_lfd(df):
    """Évolution temporelle SMOD vs LFD avec dates calendaires"""
    df_filt = df[df['annee'].between(2000, 2020)].copy()
    df_filt['type'] = df_filt['nom'].apply(
        lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
    )
    
    df_agg = df_filt.groupby(['annee', 'type']).agg(
        SMOD_moy=('SMOD_modis', 'mean'),
        LFD_moy=('LFD_nival', 'mean')
    ).reset_index()
    
    df_agg['SMOD_date'] = df_agg['SMOD_moy'].apply(jour_nival_to_date)
    df_agg['LFD_date'] = df_agg['LFD_moy'].apply(jour_nival_to_date)
    
    fig = go.Figure()
    df_noh = df_agg[df_agg['type'] == 'NOHEDES']
    
    fig.add_trace(go.Scatter(
        x=df_noh['annee'], y=df_noh['SMOD_moy'],
        mode='lines+markers', name='SMOD NOHEDES',
        line=dict(color=COLOR_NOHEDES, width=3),
        marker=dict(size=10, symbol='circle'),
        text=df_noh['SMOD_date'],
        hovertemplate='<b>NOHEDES - SMOD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_noh['annee'], y=df_noh['LFD_moy'],
        mode='lines+markers', name='LFD NOHEDES',
        line=dict(color=COLOR_NOHEDES, width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        text=df_noh['LFD_date'],
        hovertemplate='<b>NOHEDES - LFD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    df_aut = df_agg[df_agg['type'] == 'Autres sites']
    
    fig.add_trace(go.Scatter(
        x=df_aut['annee'], y=df_aut['SMOD_moy'],
        mode='lines+markers', name='SMOD moyenne 8 sites',
        line=dict(color=COLOR_AUTRES, width=3),
        marker=dict(size=10, symbol='circle'),
        text=df_aut['SMOD_date'],
        hovertemplate='<b>Autres - SMOD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_aut['annee'], y=df_aut['LFD_moy'],
        mode='lines+markers', name='LFD moyenne 8 sites',
        line=dict(color=COLOR_AUTRES, width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        text=df_aut['LFD_date'],
        hovertemplate='<b>Autres - LFD</b><br>Année %{x}<br>Jour nival : %{y:.0f}<br>Date : %{text}<extra></extra>'
    ))
    
    y_lignes = [91, 181, 273]
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
        height=550, template='plotly_white',
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
    
    df_filt['saison_SMOD'] = df_filt['SMOD_modis'].apply(jour_nival_to_saison)
    df_filt['saison_LFD'] = df_filt['LFD_nival'].apply(jour_nival_to_saison)
    
    saisons_ordre = ['Automne (SON)', 'Hiver (DJF)', 'Printemps (MAM)', 'Été (JJA)']
    
    smod_counts = df_filt.groupby(['type', 'saison_SMOD']).size().reset_index(name='n')
    lfd_counts = df_filt.groupby(['type', 'saison_LFD']).size().reset_index(name='n')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SMOD : dans quelle saison ?",
                          "LFD : dans quelle saison ?")
    )
    
    for type_site, couleur in [('NOHEDES', COLOR_NOHEDES), ('Autres sites', COLOR_AUTRES)]:
        df_type = smod_counts[smod_counts['type'] == type_site].copy()
        dict_saisons = dict(zip(df_type['saison_SMOD'], df_type['n']))
        x_vals = saisons_ordre
        y_vals = [dict_saisons.get(s, 0) for s in saisons_ordre]
        
        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals, name=type_site,
                marker_color=couleur,
                text=y_vals, textposition='outside',
                showlegend=True, legendgroup=type_site
            ),
            row=1, col=1
        )
    
    for type_site, couleur in [('NOHEDES', COLOR_NOHEDES), ('Autres sites', COLOR_AUTRES)]:
        df_type = lfd_counts[lfd_counts['type'] == type_site].copy()
        dict_saisons = dict(zip(df_type['saison_LFD'], df_type['n']))
        x_vals = saisons_ordre
        y_vals = [dict_saisons.get(s, 0) for s in saisons_ordre]
        
        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals, name=type_site,
                marker_color=couleur,
                text=y_vals, textposition='outside',
                showlegend=False, legendgroup=type_site
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=dict(
            text="<b>Distribution saisonnière des événements</b>",
            x=0.5, xanchor='center'
        ),
        height=550,
        margin=dict(t=120, b=80, l=60, r=60),
        template='plotly_white', barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2,
                      xanchor='center', x=0.5)
    )
    
    fig.update_xaxes(tickangle=-15)
    max_smod = smod_counts['n'].max() if len(smod_counts) > 0 else 0
    max_lfd = lfd_counts['n'].max() if len(lfd_counts) > 0 else 0
    max_global = max(max_smod, max_lfd, 1)
    fig.update_yaxes(title_text="Nombre d'années",
                       range=[0, max_global * 1.3 + 2])
    
    return fig


# ═══════════════════════════════════════════════════════
# FONCTIONS
# ═══════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    df = pd.read_csv("df_enrichi_v4.csv")
    df = df[(df['annee'] >= 2000) & (df['annee'] <= 2020)].copy()
    return df


def calculate_pca_mahalanobis(df, annees, variables):
    """PCA + Mahalanobis + ellipse 95%."""
    df_periode = df[df['annee'].isin(annees)].copy()
    if len(df_periode) == 0 or 'NOHEDES' not in df_periode['nom'].values:
        return None
    
    df_moy = df_periode.groupby('nom')[variables].mean().reset_index()
    df_moy = df_moy.dropna()
    
    if len(df_moy) < 3 or 'NOHEDES' not in df_moy['nom'].values:
        return None
    
    X = df_moy[variables].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    n_comp = min(len(variables), len(X_std) - 1)
    pca = SklearnPCA(n_components=n_comp)
    coords = pca.fit_transform(X_std)
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100 if n_comp >= 2 else 0
    var_pc12 = var_pc1 + var_pc2
    
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f'PC{i+1}' for i in range(n_comp)],
        index=variables
    )
    
    idx_noh_pos = list(df_moy['nom']).index('NOHEDES')
    coord_noh = coords[idx_noh_pos, :2]
    coords_autres = np.delete(coords, idx_noh_pos, axis=0)[:, :2]
    noms_autres = df_moy['nom'].drop(df_moy.index[idx_noh_pos]).values
    
    centre_autres = coords_autres.mean(axis=0)
    cov_autres = np.cov(coords_autres.T)
    
    try:
        inv_cov = np.linalg.inv(cov_autres)
        diff = coord_noh - centre_autres
        D2 = float(diff @ inv_cov @ diff)
        D = np.sqrt(D2)
    except np.linalg.LinAlgError:
        D2 = np.nan
        D = np.nan
    
    seuil_chi2 = chi2.ppf(0.95, df=2)
    p_value = 1 - chi2.cdf(D2, df=2) if not np.isnan(D2) else np.nan
    dans_ellipse = D2 <= seuil_chi2 if not np.isnan(D2) else True
    
    return {
        'coords': coords[:, :2], 'coord_noh': coord_noh,
        'coords_autres': coords_autres, 'noms_autres': noms_autres,
        'centre_autres': centre_autres, 'cov_autres': cov_autres,
        'D2': D2, 'D': D, 'seuil_chi2': seuil_chi2,
        'p_value': p_value, 'dans_ellipse': dans_ellipse,
        'loadings': loadings, 'var_pc1': var_pc1, 'var_pc2': var_pc2,
        'var_pc12': var_pc12
    }


def calculate_zscores_per_variable(df, annees, variables):
    df_periode = df[df['annee'].isin(annees)].copy()
    if len(df_periode) == 0 or 'NOHEDES' not in df_periode['nom'].values:
        return pd.DataFrame()
    
    df_moy = df_periode.groupby('nom')[variables].mean().reset_index()
    df_moy = df_moy.dropna()
    if 'NOHEDES' not in df_moy['nom'].values:
        return pd.DataFrame()
    
    noh_vals = df_moy[df_moy['nom'] == 'NOHEDES'][variables].iloc[0]
    autres_vals = df_moy[df_moy['nom'] != 'NOHEDES'][variables]
    
    resultats = []
    for var in variables:
        moy = autres_vals[var].mean()
        sd = autres_vals[var].std()
        z = (noh_vals[var] - moy) / sd if sd > 0 else 0
        
        resultats.append({
            'variable': var, 'nom_lisible': VARS_INFO.get(var, var),
            'noh': noh_vals[var], 'moy_autres': moy, 'sd_autres': sd,
            'z_score': z, 'abs_z': abs(z)
        })
    return pd.DataFrame(resultats).sort_values('abs_z', ascending=False)


def plot_zscores_barres(zscores_df, titre_suffixe=""):
    couleurs = []
    for z in zscores_df['abs_z']:
        if z < 1: couleurs.append('#4CAF50')
        elif z < 2: couleurs.append('#FF9800')
        else: couleurs.append('#C62828')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=zscores_df['nom_lisible'], x=zscores_df['z_score'],
        orientation='h', marker_color=couleurs,
        text=zscores_df['z_score'].apply(lambda x: f"{x:+.2f}"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Z-score : %{x:+.2f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_color='black', line_width=1)
    fig.add_vline(x=1, line_color='orange', line_width=1, line_dash='dot')
    fig.add_vline(x=-1, line_color='orange', line_width=1, line_dash='dot')
    fig.add_vline(x=2, line_color='red', line_width=1, line_dash='dash')
    fig.add_vline(x=-2, line_color='red', line_width=1, line_dash='dash')
    
    fig.update_layout(
        title=f"Z-scores par variable — {titre_suffixe}",
        xaxis_title='Z-score', yaxis_title='',
        height=500, template='plotly_white', margin=dict(l=250)
    )
    return fig


def plot_pca_with_ellipse(result, periode_label):
    """PCA avec ellipse Mahalanobis + flèches variables."""
    fig = go.Figure()
    
    # Ellipse 95%
    if result['cov_autres'] is not None and not np.isnan(result['D2']):
        eigenvalues, eigenvectors = np.linalg.eigh(result['cov_autres'])
        angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        chi2_val = result['seuil_chi2']
        a = np.sqrt(chi2_val * eigenvalues[1])
        b = np.sqrt(chi2_val * eigenvalues[0])
        
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        ellipse_rot = R @ np.array([ellipse_x, ellipse_y])
        
        fig.add_trace(go.Scatter(
            x=ellipse_rot[0] + result['centre_autres'][0],
            y=ellipse_rot[1] + result['centre_autres'][1],
            mode='lines', fill='toself', fillcolor=COLOR_ELLIPSE,
            line=dict(color=COLOR_ELLIPSE_LINE, width=2, dash='dash'),
            name='Ellipse 95%', hoverinfo='skip'
        ))
    
    # Loadings (flèches)
    loadings = result['loadings']
    max_coord = max(np.abs(result['coords']).max(), 1)
    max_load = np.abs(loadings.iloc[:, :2].values).max()
    scale = max_coord * 0.7 / max_load if max_load > 0 else 1
    
    for var in loadings.index:
        pc1_load = loadings.loc[var, 'PC1'] * scale
        pc2_load = loadings.loc[var, 'PC2'] * scale if 'PC2' in loadings.columns else 0
        
        fig.add_annotation(
            x=pc1_load, y=pc2_load, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor=COLOR_ARROW, showarrow=True
        )
        fig.add_trace(go.Scatter(
            x=[pc1_load * 1.15], y=[pc2_load * 1.15],
            mode='text', text=[VARS_INFO.get(var, var)],
            textfont=dict(size=9, color='#E65100'),
            textposition='middle center',
            showlegend=False, hoverinfo='skip'
        ))
    
    # 8 autres sites
    fig.add_trace(go.Scatter(
        x=result['coords_autres'][:, 0], y=result['coords_autres'][:, 1],
        mode='markers+text', name='Autres sites',
        marker=dict(size=14, color=COLOR_AUTRES,
                     line=dict(color='white', width=2)),
        text=result['noms_autres'], textposition='top center',
        textfont=dict(size=10),
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    # NOHEDES
    fig.add_trace(go.Scatter(
        x=[result['coord_noh'][0]], y=[result['coord_noh'][1]],
        mode='markers+text', name='NOHEDES',
        marker=dict(size=22, color=COLOR_NOHEDES, symbol='star',
                     line=dict(color='white', width=2)),
        text=['NOHEDES'], textposition='top center',
        textfont=dict(size=12, family='Arial Black'),
        hovertemplate='<b>NOHEDES</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>D² = ' + f"{result['D2']:.2f}" + '<extra></extra>'
    ))
    
    # Centre
    fig.add_trace(go.Scatter(
        x=[result['centre_autres'][0]], y=[result['centre_autres'][1]],
        mode='markers', name='Centre 8 sites',
        marker=dict(size=14, color='red', symbol='x',
                     line=dict(color='white', width=3)),
        hovertemplate='<b>Centre</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    # ─── Ligne NOHEDES → Centre ───
    fig.add_trace(go.Scatter(
        x=[result['coord_noh'][0], result['centre_autres'][0]],
        y=[result['coord_noh'][1], result['centre_autres'][1]],
        mode='lines',
        name=f"D = {result['D']:.3f}",
        line=dict(color='#FF6F00', width=3, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Annotation D au milieu de la ligne
    mid_x = (result['coord_noh'][0] + result['centre_autres'][0]) / 2
    mid_y = (result['coord_noh'][1] + result['centre_autres'][1]) / 2
    # Distance euclidienne pour affichage
    d_euclidienne = np.sqrt(
        (result['coord_noh'][0] - result['centre_autres'][0])**2 +
        (result['coord_noh'][1] - result['centre_autres'][1])**2
    )
    fig.add_annotation(
        x=mid_x, y=mid_y,
        text=f"<b>d = {d_euclidienne:.3f}</b><br><sub>D² Mah. = {result['D2']:.2f}</sub>",
        showarrow=False,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF6F00', borderwidth=2,
        font=dict(size=10, color='#E65100')
    )
    
    fig.add_hline(y=0, line_color='lightgray', line_width=1)
    fig.add_vline(x=0, line_color='lightgray', line_width=1)
    
    verdict = "SIMILAIRE" if result['dans_ellipse'] else "ATYPIQUE"
    couleur_verdict = "#2E7D32" if result['dans_ellipse'] else "#C62828"
    
    fig.update_layout(
        title=dict(
            text=f"<b>PCA — {periode_label}</b><br>"
                 f"<sub>D² = {result['D2']:.2f} | Seuil 5% = {result['seuil_chi2']:.2f} | "
                 f"<span style='color:{couleur_verdict}'>{verdict}</span></sub>",
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title=f'PC1 ({result["var_pc1"]:.1f}%)', zeroline=False),
        yaxis=dict(title=f'PC2 ({result["var_pc2"]:.1f}%)', zeroline=False),
        height=650, template='plotly_white',
        margin=dict(t=100, b=60, l=60, r=60),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15,
                     xanchor='center', x=0.5)
    )
    return fig


def calculer_pca_et_distances(df_input):
    """Pour distance euclidienne (méthode prof)."""
    df_sites = df_input.groupby('nom')[VARIABLES_X].mean().reset_index()
    df_sites = df_sites.dropna()
    
    if 'NOHEDES' not in df_sites['nom'].values or len(df_sites) < 3:
        return None
    
    X = df_sites[VARIABLES_X].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    pca = SklearnPCA(n_components=2)
    coords = pca.fit_transform(X_std)
    var_exp = pca.explained_variance_ratio_ * 100
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    idx_noh_pos = list(df_sites['nom']).index('NOHEDES')
    coord_noh = coords[idx_noh_pos]
    coords_autres = np.delete(coords, idx_noh_pos, axis=0)
    noms_autres = df_sites['nom'].drop(df_sites.index[idx_noh_pos]).values
    
    centroide = coords_autres.mean(axis=0)
    d_noh_centroide = np.sqrt(np.sum((coord_noh - centroide)**2))
    
    distances_noh_autres = np.sqrt(np.sum((coords_autres - coord_noh)**2, axis=1))
    dmax_local = distances_noh_autres.max()
    site_le_plus_loin = noms_autres[distances_noh_autres.argmax()]
    
    ratio_global = d_noh_centroide / DMAX_REF
    ratio_local = d_noh_centroide / dmax_local
    
    return {
        'df_sites': df_sites.reset_index(drop=True),
        'coords': coords, 'coord_noh': coord_noh,
        'coords_autres': coords_autres, 'noms_autres': noms_autres,
        'centroide': centroide, 'd_noh_centroide': d_noh_centroide,
        'dmax_local': dmax_local, 'ratio_global': ratio_global,
        'ratio_local': ratio_local, 'site_le_plus_loin': site_le_plus_loin,
        'var_exp': var_exp, 'loadings': loadings
    }


def faire_biplot_pca(res, annee_label):
    """Biplot PCA avec noms lisibles."""
    fig = go.Figure()
    coords = res['coords']
    coord_noh = res['coord_noh']
    coords_autres = res['coords_autres']
    noms_autres = res['noms_autres']
    centroide = res['centroide']
    loadings = res['loadings']
    var_exp = res['var_exp']
    
    max_coord = max(np.abs(coords).max(), 1)
    max_load = np.abs(loadings).max()
    scale_factor = max_coord * 0.75 / max_load if max_load > 0 else 1
    
    for i, var in enumerate(VARIABLES_X):
        var_lisible = VARS_INFO.get(var, var)
        
        fig.add_annotation(
            x=loadings[i, 0] * scale_factor,
            y=loadings[i, 1] * scale_factor,
            ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor='rgba(255, 140, 0, 0.85)',
            showarrow=True
        )
        
        fig.add_trace(go.Scatter(
            x=[loadings[i, 0] * scale_factor * 1.15],
            y=[loadings[i, 1] * scale_factor * 1.15],
            mode='text', text=[var_lisible],
            textfont=dict(size=10, color='#E65100'),
            textposition='middle center',
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter(
        x=coords_autres[:, 0], y=coords_autres[:, 1],
        mode='markers+text', name='Autres sites (8)',
        marker=dict(size=14, color=COLOR_AUTRES,
                     line=dict(color='white', width=2)),
        text=noms_autres, textposition='top center',
        textfont=dict(size=10),
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[coord_noh[0]], y=[coord_noh[1]],
        mode='markers+text', name='NOHEDES',
        marker=dict(size=20, color=COLOR_NOHEDES, symbol='star',
                     line=dict(color='white', width=2)),
        text=['NOHEDES'], textposition='top center',
        textfont=dict(size=12, family='Arial Black'),
        hovertemplate='<b>NOHEDES</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[centroide[0]], y=[centroide[1]],
        mode='markers+text', name='Centroïde (8 sites)',
        marker=dict(size=18, color='#E53935', symbol='x',
                     line=dict(color='white', width=3)),
        text=['⭐ Centroïde'], textposition='bottom center',
        textfont=dict(size=11, family='Arial Black'),
        hovertemplate='<b>Centroïde</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[coord_noh[0], centroide[0]],
        y=[coord_noh[1], centroide[1]],
        mode='lines', name=f"d = {res['d_noh_centroide']:.3f}",
        line=dict(color='#FF6F00', width=3, dash='dash'),
        hoverinfo='skip'
    ))
    
    mid_x = (coord_noh[0] + centroide[0]) / 2
    mid_y = (coord_noh[1] + centroide[1]) / 2
    fig.add_annotation(
        x=mid_x, y=mid_y,
        text=f"<b>d = {res['d_noh_centroide']:.3f}</b>",
        showarrow=False, bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#FF6F00', borderwidth=2,
        font=dict(size=11, color='#E65100')
    )
    
    fig.add_hline(y=0, line_color='lightgray', line_width=1)
    fig.add_vline(x=0, line_color='lightgray', line_width=1)
    
    fig.update_layout(
        title=dict(
            text=f"<b>PCA — {annee_label}</b><br>"
                 f"<sub>Variance expliquée : PC1 = {var_exp[0]:.1f}% | PC2 = {var_exp[1]:.1f}%</sub>",
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title=f'PC1 ({var_exp[0]:.1f}%)', zeroline=False),
        yaxis=dict(title=f'PC2 ({var_exp[1]:.1f}%)', zeroline=False),
        height=650, template='plotly_white',
        margin=dict(t=100, b=60, l=60, r=60),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15,
                     xanchor='center', x=0.5),
        hovermode='closest'
    )
    return fig


# ═══════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════

try:
    df = charger_donnees()
except FileNotFoundError:
    st.error("❌ Fichier `df_enrichi_v4.csv` introuvable.")
    st.stop()

VARS_INFO = {k: v for k, v in VARS_INFO.items() if k in df.columns}
VARIABLES_X = list(VARS_INFO.keys())

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════

st.title("🌸 NOHEDES — Analyse de l'atypicité climatique")

st.markdown("""
<div class="histoire-box">
Cette application analyse la <strong>spécificité climatique de NOHEDES</strong> 
par rapport aux 8 autres sites Pyrénéens fleurissants sur 7 variables climatiques.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Présentation",
    "📊 PCA & Mahalanobis",
    "🌐 Niche NMI",
    "🌨️ SMOD vs LFD",
    "🧪 Test de Tukey",
    "📚 Références"
])

# ─────────────────────────────────────────────
# TAB 2 — PCA & MAHALANOBIS
# ─────────────────────────────────────────────
with tab2:
    st.markdown("# 📊 PCA & Distance de Mahalanobis")
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        vars_select = st.multiselect(
            "Variables à analyser",
            options=list(VARS_INFO.keys()),
            default=list(VARS_INFO.keys()),
            format_func=lambda x: VARS_INFO.get(x, x),
            key='mah_vars'
        )
    
    with col_c2:
        fenetre_choisie = st.selectbox(
            "Fenêtre temporelle",
            options=list(FENETRES.keys()),
            index=2,
            key='mah_fen'
        )
    
    if len(vars_select) < 2:
        st.warning("⚠️ Sélectionne au moins 2 variables.")
    else:
        annees = FENETRES[fenetre_choisie]
        result = calculate_pca_mahalanobis(df, annees, vars_select)
        
        if result is None:
            st.error("⚠ Pas assez de données valides pour la PCA")
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Distance D²", f"{result['D2']:.2f}",
                            delta=f"Seuil 5% : {result['seuil_chi2']:.2f}",
                            delta_color="inverse")
            with col_m2:
                st.metric("Distance D", f"{result['D']:.2f}",
                            delta=f"Seuil 5% : {np.sqrt(result['seuil_chi2']):.2f}",
                            delta_color="inverse")
            with col_m3:
                st.metric("p-value", f"{result['p_value']:.4f}",
                            delta="< 0.05 = atypique", delta_color="off")
            with col_m4:
                if result['dans_ellipse']:
                    st.metric("Verdict", "✓ SIMILAIRE")
                    st.success("NOHEDES dans l'ellipse 95%")
                else:
                    st.metric("Verdict", "⚠ ATYPIQUE")
                    st.error("NOHEDES hors de l'ellipse 95%")
            
            col_g1, col_g2 = st.columns([2, 1])
            
            with col_g1:
                fig_pca = plot_pca_with_ellipse(result, fenetre_choisie)
                st.plotly_chart(fig_pca, use_container_width=True,
                                 key='pca_mah_main')
            
            with col_g2:
                st.markdown("### 📊 Contributions des variables")
                
                loadings = result['loadings'].copy()
                loadings['Variable'] = loadings.index.map(VARS_INFO)
                loadings['|PC1|'] = loadings['PC1'].abs()
                loadings = loadings.sort_values('|PC1|', ascending=False)
                
                df_display = loadings[['Variable', 'PC1', 'PC2']].copy()
                df_display['PC1'] = df_display['PC1'].apply(lambda x: f"{x:+.3f}")
                df_display['PC2'] = df_display['PC2'].apply(lambda x: f"{x:+.3f}")
                
                st.dataframe(df_display, hide_index=True,
                              use_container_width=True, height=400)
                
                st.caption(
                    f"📊 Variance expliquée : "
                    f"**{result['var_pc12']:.1f}%** "
                    f"(PC1: {result['var_pc1']:.1f}% + PC2: {result['var_pc2']:.1f}%)"
                )
            
            with st.expander("ℹ Comment interpréter ces résultats ?"):
                st.markdown(
                    f"""
                    ### 📐 Distance Mahalanobis
                    
                    La **distance de Mahalanobis (D²)** mesure l'éloignement de NOHEDES 
                    par rapport au centre du groupe des 8 sites fleurissants, en tenant 
                    compte de la dispersion et de la corrélation des variables.
                    
                    ### 🎯 Seuil de décision
                    
                    - **D² = {result['D2']:.2f}**
                    - **Seuil χ²(2) à 5% = {result['seuil_chi2']:.2f}**
                    - **Conclusion :** {'NOHEDES est ATYPIQUE (D² > seuil)' if not result['dans_ellipse'] else 'NOHEDES est SIMILAIRE (D² ≤ seuil)'}
                    
                    ### 🔵 Ellipse de confiance 95%
                    
                    L'ellipse représente la zone où **95% des sites fleurissants** 
                    devraient se trouver statistiquement. Si NOHEDES est en dehors, 
                    il diffère significativement (p < 0.05).
                    """
                )
            
            st.markdown("---")
            st.subheader("📊 Z-scores par variable — Où NOHEDES diffère-t-il ?")
            
            if result['dans_ellipse']:
                st.success("✅ NOHEDES est **SIMILAIRE** globalement.")
            else:
                st.warning("⚠ NOHEDES est **ATYPIQUE** globalement.")
            
            zscores_df = calculate_zscores_per_variable(df, annees, vars_select)
            
            if len(zscores_df) > 0:
                fig_zscores = plot_zscores_barres(
                    zscores_df, titre_suffixe=f"Fenêtre {fenetre_choisie}"
                )
                st.plotly_chart(fig_zscores, use_container_width=True,
                                 key='zscores_bar')
                
                col_l1, col_l2, col_l3 = st.columns(3)
                col_l1.markdown("🟢 **|z| < 1** : NOHEDES similaire")
                col_l2.markdown("🟠 **|z| 1-2** : Modérément différent")
                col_l3.markdown("🔴 **|z| ≥ 2** : Très différent (p<0.05)")

# ─────────────────────────────────────────────
# TAB 3 — NICHE NMI
# ─────────────────────────────────────────────
with tab3:
    st.markdown("# 🌐 Niche Margin Index (NMI)")
    
    st.markdown("""
    <div class="histoire-box">
    <strong>📊 Méthode NMI</strong> (d'après Broennimann et al., 
    Nature Communications 2021) :<br>
    1. Estimation de la <strong>niche climatique</strong> des 8 sites 
    fleurissants (densité de kernel 2D)<br>
    2. Contour = <strong>marge à 95%</strong> (contour vert)<br>
    3. NMI = distance à la marge / dmax<br>
    4. <strong>NMI positif</strong> → DANS la niche ✅<br>
    5. <strong>NMI négatif</strong> → HORS la niche ⚠️
    </div>
    """, unsafe_allow_html=True)
    
    # Calcul (caché)
    with st.spinner("Calcul de la niche et des NMI..."):
        nmi_data = calculer_niche_et_NMI()
    
    df_NMI = nmi_data['df_NMI']
    
    st.divider()
    
    # ══════════════════════════════════════════
    # SECTION 1 — GRAPHIQUE STATIQUE
    # ══════════════════════════════════════════
    st.markdown("## 🗺️ Carte de la niche + Positions NOHEDES (toutes les années)")
    
    # Graphique 1 : Heatmap NMI + 8 sites + toutes années NOHEDES
    fig_niche = go.Figure()
    
    # Heatmap NMI (fond)
    fig_niche.add_trace(go.Contour(
        x=nmi_data['XX'][0, :],
        y=nmi_data['YY'][:, 0],
        z=nmi_data['grid_NMI'],
        colorscale=[
            [0, '#FFB74D'],   # Orange = HORS
            [0.5, 'white'],   # Blanc = marge
            [1, '#1976D2']    # Bleu = DANS
        ],
        contours=dict(showlines=False),
        colorbar=dict(title='NMI', x=1.02),
        opacity=0.7,
        showscale=True,
        hoverinfo='skip'
    ))
    
    # Contour de la niche (marge à 95%)
    fig_niche.add_trace(go.Contour(
        x=nmi_data['XX'][0, :],
        y=nmi_data['YY'][:, 0],
        z=nmi_data['grid_densites'],
        contours=dict(
            start=nmi_data['seuil_densite'],
            end=nmi_data['seuil_densite'],
            coloring='none',
            showlines=True
        ),
        line=dict(color='#2E7D32', width=3),
        showscale=False,
        name='Marge 95%',
        hoverinfo='skip'
    ))
    
    # 8 sites fleurissants (positions moyennes)
    fig_niche.add_trace(go.Scatter(
        x=nmi_data['coords_8sites'][:, 0],
        y=nmi_data['coords_8sites'][:, 1],
        mode='markers+text',
        name='8 sites fleurissants',
        marker=dict(size=15, color='purple', symbol='triangle-up',
                     line=dict(color='white', width=2)),
        text=nmi_data['noms_8sites'],
        textposition='top center',
        textfont=dict(size=10, color='purple'),
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))
    
    # NOHEDES par année (toutes années)
    df_dans = df_NMI[df_NMI['statut'] == 'DANS niche']
    df_hors = df_NMI[df_NMI['statut'] == 'HORS niche']
    
    if len(df_dans) > 0:
        fig_niche.add_trace(go.Scatter(
            x=df_dans['PC1_NOHEDES'],
            y=df_dans['PC2_NOHEDES'],
            mode='markers+text',
            name='NOHEDES DANS niche',
            marker=dict(size=10, color='#2E7D32',
                         line=dict(color='white', width=1)),
            text=df_dans['annee'].astype(str),
            textposition='top center',
            textfont=dict(size=8, color='#1B5E20'),
            hovertemplate='<b>%{text}</b><br>NMI: ' + df_dans['NMI'].apply(lambda x: f"{x:+.3f}") + '<extra></extra>'
        ))
    
    if len(df_hors) > 0:
        fig_niche.add_trace(go.Scatter(
            x=df_hors['PC1_NOHEDES'],
            y=df_hors['PC2_NOHEDES'],
            mode='markers+text',
            name='NOHEDES HORS niche',
            marker=dict(size=10, color='#C62828',
                         line=dict(color='white', width=1)),
            text=df_hors['annee'].astype(str),
            textposition='top center',
            textfont=dict(size=8, color='#B71C1C'),
            hovertemplate='<b>%{text}</b><br>NMI: ' + df_hors['NMI'].apply(lambda x: f"{x:+.3f}") + '<extra></extra>'
        ))
    
    fig_niche.update_layout(
        title=dict(
            text="<b>Niche climatique des 8 sites fleurissants + positions NOHEDES</b><br>"
                 "<sub>Contour vert = marge de la niche (95%) | Bleu = DANS | Orange = HORS</sub>",
            x=0.5, xanchor='center'
        ),
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=650, template='plotly_white',
        margin=dict(t=100, b=60, l=60, r=100),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15,
                     xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig_niche, use_container_width=True, key='niche_statique')
    
    st.divider()
    
    # ══════════════════════════════════════════
    # SECTION 2 — ANIMATION VIDÉO (Play/Pause)
    # ══════════════════════════════════════════
    st.markdown("## 🎬 Position NOHEDES année par année (animation)")
    st.caption("▶️ Bouton Play/Pause en bas du graphique")
    
    annees_dispo_nmi = sorted(df_NMI['annee'].unique())
    
    # ⭐ TRACES FIXES (heatmap, contour, 8 sites) — indices 0, 1, 2
    # Ces traces NE FONT PAS partie des frames
    fixed_traces = [
        # Trace 0 : Heatmap NMI (fond)
        go.Contour(
            x=nmi_data['XX'][0, :], y=nmi_data['YY'][:, 0],
            z=nmi_data['grid_NMI'],
            colorscale=[[0, '#FFB74D'], [0.5, 'white'], [1, '#1976D2']],
            contours=dict(showlines=False),
            colorbar=dict(title='NMI', x=1.02),
            opacity=0.7, showscale=True, hoverinfo='skip'
        ),
        # Trace 1 : Contour marge 95%
        go.Contour(
            x=nmi_data['XX'][0, :], y=nmi_data['YY'][:, 0],
            z=nmi_data['grid_densites'],
            contours=dict(
                start=nmi_data['seuil_densite'],
                end=nmi_data['seuil_densite'],
                coloring='none', showlines=True
            ),
            line=dict(color='#2E7D32', width=3),
            showscale=False, hoverinfo='skip'
        ),
        # Trace 2 : 8 sites (fixes)
        go.Scatter(
            x=nmi_data['coords_8sites'][:, 0],
            y=nmi_data['coords_8sites'][:, 1],
            mode='markers+text',
            marker=dict(size=15, color='purple', symbol='triangle-up',
                         line=dict(color='white', width=2)),
            text=nmi_data['noms_8sites'],
            textposition='top center',
            textfont=dict(size=10, color='purple'),
            name='8 sites'
        )
    ]
    
    # Trace initiale de NOHEDES (index 3)
    row_init = df_NMI.iloc[0]
    couleur_init = '#2E7D32' if row_init['statut'] == 'DANS niche' else '#C62828'
    
    initial_noh = go.Scatter(
        x=[row_init['PC1_NOHEDES']], y=[row_init['PC2_NOHEDES']],
        mode='markers+text',
        marker=dict(size=25, color=couleur_init, symbol='star',
                     line=dict(color='white', width=3)),
        text=[f"NOHEDES_{row_init['annee']}"],
        textposition='top center',
        textfont=dict(size=13, color=couleur_init, family='Arial Black'),
        name='NOHEDES'
    )
    
    # ⭐ FRAMES : seule la trace NOHEDES change (index 3)
    frames = []
    for an in annees_dispo_nmi:
        row = df_NMI[df_NMI['annee'] == an].iloc[0]
        couleur_pt = '#2E7D32' if row['statut'] == 'DANS niche' else '#C62828'
        
        frames.append(go.Frame(
            data=[
                # Seulement le point NOHEDES change
                go.Scatter(
                    x=[row['PC1_NOHEDES']], y=[row['PC2_NOHEDES']],
                    mode='markers+text',
                    marker=dict(size=25, color=couleur_pt, symbol='star',
                                 line=dict(color='white', width=3)),
                    text=[f"NOHEDES_{an}"],
                    textposition='top center',
                    textfont=dict(size=13, color=couleur_pt, family='Arial Black')
                )
            ],
            # ⭐ IMPORTANT : traces=[3] indique que seule la trace 3 est remplacée
            traces=[3],
            name=str(an),
            layout=go.Layout(
                title=f"<b>NOHEDES en {an}</b> — NMI = {row['NMI']:+.3f} — {row['statut']}"
            )
        ))
    
    # Figure avec traces fixes + trace initiale NOHEDES
    fig_anim = go.Figure(
        data=fixed_traces + [initial_noh],
        frames=frames
    )
    
    fig_anim.update_layout(
        title=f"<b>NOHEDES en {row_init['annee']}</b> — NMI = {row_init['NMI']:+.3f} — {row_init['statut']}",
        xaxis_title='PC1', yaxis_title='PC2',
        height=650, template='plotly_white',
        margin=dict(t=100, b=100, l=60, r=100),
        showlegend=False,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=-0.05, x=0.5,
            xanchor='center', yanchor='top',
            direction='left',
            buttons=[
                dict(label='▶️ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=800, redraw=True),
                                       fromcurrent=True,
                                       transition=dict(duration=300))]),
                dict(label='⏸️ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode='immediate',
                                          transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Année : ', font=dict(size=14)),
            pad=dict(t=50),
            steps=[dict(method='animate', label=str(an),
                          args=[[str(an)],
                                dict(mode='immediate',
                                     frame=dict(duration=500, redraw=True),
                                     transition=dict(duration=300))])
                   for an in annees_dispo_nmi]
        )]
    )
    
    st.plotly_chart(fig_anim, use_container_width=True, key='niche_animation')
    
    st.divider()
    
    st.markdown("## 📈 Évolution du NMI de NOHEDES (2000-2020)")
    
    fig_evo_nmi = go.Figure()
    
    # Ligne principale
    fig_evo_nmi.add_trace(go.Scatter(
        x=df_NMI['annee'],
        y=df_NMI['NMI'],
        mode='lines',
        line=dict(color='#7E3AC8', width=2.5),
        name='NMI',
        hoverinfo='skip'
    ))
    
    # Points colorés selon dedans/dehors
    for statut, couleur, label_stat in [
        ('DANS niche', '#2E7D32', 'DANS niche'),
        ('HORS niche', '#C62828', 'HORS niche')
    ]:
        sub = df_NMI[df_NMI['statut'] == statut]
        if len(sub) > 0:
            fig_evo_nmi.add_trace(go.Scatter(
                x=sub['annee'],
                y=sub['NMI'],
                mode='markers+text',
                name=label_stat,
                marker=dict(size=12, color=couleur,
                              line=dict(color='white', width=2)),
                text=sub['NMI'].apply(lambda x: f"{x:+.2f}"),
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate='<b>Année %{x}</b><br>NMI: %{y:+.3f}<extra></extra>'
            ))
    
    # Ligne 0 (marge)
    fig_evo_nmi.add_hline(y=0, line_dash='dash', line_color='red',
                            line_width=2,
                            annotation_text='Marge de la niche (NMI = 0)',
                            annotation_position='right')
    
    fig_evo_nmi.update_layout(
        title=dict(
            text="<b>Évolution du NMI de NOHEDES (2000-2020)</b><br>"
                 "<sub>Positif = DANS la niche | Négatif = HORS</sub>",
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title='Année', dtick=2),
        yaxis_title='NMI',
        height=500, template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2,
                     xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig_evo_nmi, use_container_width=True, key='evo_nmi')
    
    # Statistiques
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    n_dans = len(df_NMI[df_NMI['statut'] == 'DANS niche'])
    n_hors = len(df_NMI[df_NMI['statut'] == 'HORS niche'])
    n_tot = len(df_NMI)
    
    col_s1.metric("✅ DANS niche", f"{n_dans}/{n_tot}",
                    delta=f"{100*n_dans/n_tot:.1f}%")
    col_s2.metric("❌ HORS niche", f"{n_hors}/{n_tot}",
                    delta=f"{100*n_hors/n_tot:.1f}%")
    col_s3.metric("NMI min", f"{df_NMI['NMI'].min():+.3f}",
                    delta=f"an {df_NMI.loc[df_NMI['NMI'].idxmin(), 'annee']}")
    col_s4.metric("NMI max", f"{df_NMI['NMI'].max():+.3f}",
                    delta=f"an {df_NMI.loc[df_NMI['NMI'].idxmax(), 'annee']}")

    st.divider()
    
    # ══════════════════════════════════════════
    # SECTION 3 — ANALYSE PAR VARIABLE (année par année)
    # ══════════════════════════════════════════
    st.markdown("## 📊 Analyse statistique par année")
    st.caption("Sélectionne DANS ou HORS niche pour voir l'analyse annuelle par variable")
    
    choix_type = st.radio(
        "Type d'années",
        options=["NOHEDES HORS niche", "NOHEDES DANS niche"],
        horizontal=True,
        key='choix_dans_hors'
    )
    
    annees_HORS = df_NMI[df_NMI['statut'] == 'HORS niche']['annee'].tolist()
    annees_DANS = df_NMI[df_NMI['statut'] == 'DANS niche']['annee'].tolist()
    
    if choix_type == "NOHEDES HORS niche":
        annees_liste = annees_HORS
    else:
        annees_liste = annees_DANS
    
    st.info(f"📅 {len(annees_liste)} années : {', '.join(str(a) for a in annees_liste)}")
    
    # Tableau consolidé pour toutes les années
    all_stats = []
    for an in annees_liste:
        df_an = df[df['annee'] == an].copy()
        df_an['groupe'] = df_an['nom'].apply(
            lambda n: 'NOHEDES' if n == 'NOHEDES' else 'Autres'
        )
        
        for var in VARIABLES_X:
            noh_vals = df_an[df_an['groupe'] == 'NOHEDES'][var].dropna()
            aut_vals = df_an[df_an['groupe'] == 'Autres'][var].dropna()
            
            if len(noh_vals) > 0 and len(aut_vals) > 1:
                noh_val = noh_vals.iloc[0]
                aut_moy = aut_vals.mean()
                ecart = noh_val - aut_moy
                
                try:
                    t_stat, p_val = stats.ttest_ind(noh_vals, aut_vals,
                                                      equal_var=True)
                except Exception:
                    p_val = np.nan
                
                all_stats.append({
                    'Année': an,
                    'Variable': VARS_INFO.get(var, var),
                    'NOHEDES': round(noh_val, 2),
                    'Autres (moy)': round(aut_moy, 2),
                    'Écart': round(ecart, 2),
                    'p-value': round(p_val, 4) if not np.isnan(p_val) else '-'
                })
    
    df_all_stats = pd.DataFrame(all_stats)
    
    # Sélecteur année
    annee_choix = st.selectbox(
        "📅 Voir année précise",
        options=annees_liste,
        key='annee_choix_stat'
    )
    
    df_annee_stat = df_all_stats[df_all_stats['Année'] == annee_choix].drop(columns=['Année'])
    st.dataframe(df_annee_stat, use_container_width=True, hide_index=True)
    
# ─────────────────────────────────────────────
# TAB 5 — TEST DE TUKEY
# ─────────────────────────────────────────────
with tab5:
    st.markdown("# 🧪 Test de Tukey HSD")
    
    # Sélecteur année
    annees_dispo_tuk = sorted(df['annee'].unique())
    annee_tukey = st.selectbox(
        "📅 Année",
        options=annees_dispo_tuk,
        index=annees_dispo_tuk.index(2010) if 2010 in annees_dispo_tuk else 0,
        key='annee_tukey'
    )
    
    df_tuk = df[df['annee'] == annee_tukey].copy()
    df_tuk['groupe'] = df_tuk['nom'].apply(
        lambda n: 'NOHEDES' if n == 'NOHEDES' else 'Autres sites'
    )
    
    # Fonction Tukey pour 1 variable
    def tukey_lettres_var(df_data, var):
        sub = df_data[['groupe', var]].dropna()
        if sub['groupe'].nunique() < 2:
            return 'a', 'a', np.nan
        
        noh_vals = sub[sub['groupe'] == 'NOHEDES'][var].values
        aut_vals = sub[sub['groupe'] == 'Autres sites'][var].values
        
        if len(noh_vals) == 0 or len(aut_vals) < 2:
            return 'a', 'a', np.nan
        
        try:
            t_stat, p_val = stats.ttest_ind(noh_vals, aut_vals, equal_var=True)
            moy_noh = np.mean(noh_vals)
            moy_aut = np.mean(aut_vals)
            
            if p_val < 0.05:
                if moy_noh > moy_aut:
                    return 'b', 'a', p_val
                else:
                    return 'a', 'b', p_val
            else:
                return 'a', 'a', p_val
        except Exception:
            return 'a', 'a', np.nan
    
    # Boxplots subplot (2 lignes × 4 colonnes)
    n_vars = len(VARIABLES_X)
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))
    
    fig_tuk = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[VARS_INFO.get(v, v) for v in VARIABLES_X],
        vertical_spacing=0.15, horizontal_spacing=0.06
    )
    
    for i, var in enumerate(VARIABLES_X):
        row_i = (i // n_cols) + 1
        col_i = (i % n_cols) + 1
        
        lettre_noh, lettre_aut, pval = tukey_lettres_var(df_tuk, var)
        
        noh_data = df_tuk[df_tuk['groupe'] == 'NOHEDES'][var].dropna()
        aut_data = df_tuk[df_tuk['groupe'] == 'Autres sites'][var].dropna()
        
        if len(noh_data) > 0:
            fig_tuk.add_trace(
                go.Box(
                    y=noh_data, name='NOHEDES',
                    marker_color=COLOR_NOHEDES,
                    boxpoints='all', jitter=0.3, pointpos=0,
                    showlegend=(i == 0), legendgroup='NOHEDES'
                ),
                row=row_i, col=col_i
            )
        
        if len(aut_data) > 0:
            fig_tuk.add_trace(
                go.Box(
                    y=aut_data, name='Autres sites',
                    marker_color=COLOR_AUTRES,
                    boxpoints='all', jitter=0.3, pointpos=0,
                    showlegend=(i == 0), legendgroup='Autres'
                ),
                row=row_i, col=col_i
            )
        
        # Annotations LETTRES au-dessus
        all_vals = pd.concat([noh_data, aut_data])
        if len(all_vals) > 0:
            y_max = all_vals.max()
            y_min = all_vals.min()
            y_pos = y_max + (y_max - y_min) * 0.20 if y_max != y_min else y_max + 1
            
            fig_tuk.add_annotation(
                x='NOHEDES', y=y_pos,
                text=f"<b>{lettre_noh}</b>",
                showarrow=False,
                font=dict(size=18, color=COLOR_NOHEDES),
                row=row_i, col=col_i
            )
            fig_tuk.add_annotation(
                x='Autres sites', y=y_pos,
                text=f"<b>{lettre_aut}</b>",
                showarrow=False,
                font=dict(size=18, color=COLOR_AUTRES),
                row=row_i, col=col_i
            )
    
    fig_tuk.update_layout(
        title=f"Boxplots + Tukey — Année {annee_tukey}",
        height=350 * n_rows,
        template='plotly_white',
        margin=dict(t=100, b=60, l=60, r=60),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.05,
                     xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig_tuk, use_container_width=True,
                     key=f'tukey_{annee_tukey}')
    
    st.divider()
    
    # ══════════════════════════════════════════
    # ANALYSE : NB années où NOHEDES ≠ autres par variable
    # ══════════════════════════════════════════
    st.markdown("### 📊 Synthèse sur 21 années")
    
    @st.cache_data
    def synthese_tukey_par_variable():
        resultats = []
        df_all = df.copy()
        df_all['groupe'] = df_all['nom'].apply(
            lambda n: 'NOHEDES' if n == 'NOHEDES' else 'Autres sites'
        )
        
        for var in VARIABLES_X:
            n_differentes = 0
            p_values = []
            
            for an in sorted(df_all['annee'].unique()):
                df_an = df_all[df_all['annee'] == an]
                sub = df_an[['groupe', var]].dropna()
                
                if sub['groupe'].nunique() < 2:
                    continue
                
                noh_vals = sub[sub['groupe'] == 'NOHEDES'][var].values
                aut_vals = sub[sub['groupe'] == 'Autres sites'][var].values
                
                if len(noh_vals) == 0 or len(aut_vals) < 2:
                    continue
                
                try:
                    t_stat, pval = stats.ttest_ind(noh_vals, aut_vals,
                                                     equal_var=True)
                    if not np.isnan(pval):
                        p_values.append(pval)
                        if pval < 0.05:
                            n_differentes += 1
                except Exception:
                    continue
            
            n_annees = len(p_values)
            resultats.append({
                'Variable': VARS_INFO.get(var, var),
                'n_années': n_annees,
                'n_différentes': n_differentes,
                '% différentes': round(100 * n_differentes / n_annees, 1) if n_annees > 0 else 0,
                'p_moyenne': round(np.mean(p_values), 4) if p_values else np.nan
            })
        
        return pd.DataFrame(resultats).sort_values('n_différentes', ascending=False)
    
    df_syn = synthese_tukey_par_variable()
    st.dataframe(df_syn, use_container_width=True, hide_index=True)
    
    # Bar chart synthèse
    fig_syn = go.Figure()
    
    couleurs = ['#2E7D32' if p >= 20 else '#FFA726' if p >= 10 else '#C62828' 
                 for p in df_syn['% différentes']]
    
    fig_syn.add_trace(go.Bar(
        y=df_syn['Variable'],
        x=df_syn['n_différentes'],
        orientation='h',
        marker_color=couleurs,
        text=df_syn.apply(lambda r: f"{r['n_différentes']}/{r['n_années']} ({r['% différentes']}%)",
                            axis=1),
        textposition='outside'
    ))
    
    fig_syn.update_layout(
        title="Nombre d'années où NOHEDES diffère significativement (p < 0.05)",
        xaxis_title="Nb années différentes sur 21",
        yaxis_title="",
        height=400, template='plotly_white',
        margin=dict(l=250)
    )
    
    st.plotly_chart(fig_syn, use_container_width=True, key='syn_tukey')


# ─────────────────────────────────────────────
# TAB 4 — SMOD vs LFD
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### 🌨️ Comparaison SMOD (fin de la neige) vs LFD (dernier gel)")
    
    if 'LFD_nival' not in df.columns:
        st.error("⚠ La variable LFD_nival n'est pas disponible dans le jeu de données.")
    else:
        st.markdown(
            "Cette section compare l'évolution temporelle de deux dates "
            "clés du cycle annuel : **SMOD** (fin de l'enneigement) et "
            "**LFD** (dernier jour de gel)."
        )
        
        # ━━━ ÉVOLUTION TEMPORELLE ━━━
        st.markdown("#### 📈 Évolution annuelle (2000-2020)")
        
        fig_evolution_smod_lfd = plot_evolution_smod_lfd(df)
        st.plotly_chart(fig_evolution_smod_lfd, use_container_width=True,
                         key='smod_lfd_evolution')
        
        # Stats globales
        df_calc = df[df['annee'].between(2000, 2020)].copy()
        df_calc['type'] = df_calc['nom'].apply(
            lambda x: 'NOHEDES' if x == 'NOHEDES' else 'Autres sites'
        )
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("##### 📊 SMOD (fin enneigement)")
            noh_smod = df_calc[df_calc['type'] == 'NOHEDES']['SMOD_modis'].mean()
            aut_smod = df_calc[df_calc['type'] == 'Autres sites']['SMOD_modis'].mean()
            
            st.metric("NOHEDES",
                        f"{jour_nival_to_date(noh_smod)} (jour {noh_smod:.0f})")
            st.metric("Moyenne 8 sites",
                        f"{jour_nival_to_date(aut_smod)} (jour {aut_smod:.0f})")
            
            diff_smod = noh_smod - aut_smod
            st.caption(
                f"Différence : **{diff_smod:+.0f} jours** "
                f"({'NOHEDES PLUS TARDIF' if diff_smod > 0 else 'NOHEDES PLUS PRÉCOCE'})"
            )
        
        with col_s2:
            st.markdown("##### 🥶 LFD (dernier gel)")
            noh_lfd = df_calc[df_calc['type'] == 'NOHEDES']['LFD_nival'].mean()
            aut_lfd = df_calc[df_calc['type'] == 'Autres sites']['LFD_nival'].mean()
            
            st.metric("NOHEDES",
                        f"{jour_nival_to_date(noh_lfd)} (jour {noh_lfd:.0f})")
            st.metric("Moyenne 8 sites",
                        f"{jour_nival_to_date(aut_lfd)} (jour {aut_lfd:.0f})")
            
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
        st.plotly_chart(fig_saisons, use_container_width=True,
                         key='smod_lfd_saisons')
        
        # Tableau détail
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

# ─────────────────────────────────────────────
# TAB 1 — PRÉSENTATION
# ─────────────────────────────────────────────
with tab1:
    st.markdown("# 📖 Présentation de l'étude")
    
    st.markdown("""
    <div class="histoire-box">
    <strong>🌸 Objectif</strong> : Comprendre pourquoi <b>NOHEDES</b> (site à 1790 m, 
    dans les Pyrénées) ne voit plus fleurir <b>Delphinium montanum</b>, contrairement 
    aux 8 autres sites Pyrénéens. Nous analysons son <b>climat</b> comme cause possible.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 1 — Les 8 variables
    # ═══════════════════════════════════════════════
    st.markdown("## 🌡️ Les 8 variables climatiques analysées")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("""
        **☀️ Température air moyenne (°C)**  
        Chaleur ressentie dans l'air.
        
        **🌡️ Température sol (°C)**  
        Chaleur mesurée dans le sol (10 cm).
        
        **💧 Humidité air moyenne (%)**  
        Vapeur d'eau dans l'atmosphère.
        
        **🌧️ % précipitations**  
        Part des précipitations sur l'année.
        """)
    
    with col_v2:
        st.markdown("""
        **❄️ Fin enneigement (SMOD)**  
        Jour où la neige disparaît (satellite).
        
        **❄️ Dernier jour de neige (LFD)**  
        Jour final où il y a eu de la neige au sol.
        
        **❄️ Nombre de jours de neige (SCD)**  
        Combien de jours il y a de la neige.
        
        **❄️ Continuité du manteau neigeux**  
        La neige est-elle régulière ou pas ?
        """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 2 — Les 4 méthodes
    # ═══════════════════════════════════════════════
    st.markdown("## 🔬 Les 4 méthodes d'analyse")
    
    st.markdown("""
    ### 📊 1. PCA & Mahalanobis
    Combine les 8 variables pour créer un espace simplifié à 2 dimensions. 
    La **distance de Mahalanobis** mesure statistiquement si NOHEDES est 
    différent des 8 autres sites. 
    → *Si NOHEDES est HORS de l'ellipse 95% : il est atypique.*
    """)
    
    st.markdown("""
    ### 🌐 2. Niche NMI (Niche Margin Index)
    Estime la **"niche climatique"** des sites fleurissants (leur enveloppe). 
    Le NMI mesure si NOHEDES est **DANS** ou **HORS** de cette niche.  
    → *NMI positif = NOHEDES est comme les autres. NMI négatif = NOHEDES est différent.*
    """)
    
    st.markdown("""
    ### 🌨️ 3. SMOD vs LFD
    Compare deux indicateurs neigeux importants : 
    la **fin d'enneigement** (SMOD) et le **dernier jour de neige** (LFD).  
    → *Aide à comprendre le cycle nival de chaque site.*
    """)
    
    st.markdown("""
    ### 🧪 4. Test de Tukey HSD
    Test statistique qui vérifie **variable par variable** si NOHEDES 
    diffère significativement des 8 autres sites.  
    → *Résultat = lettres (a/a = identique, a/b = différent).*
    """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 3 — Lecture
    # ═══════════════════════════════════════════════
    st.markdown("## 💡 Comment lire cette application")
    
    st.markdown("""
    Chaque onglet applique une **méthode différente** aux mêmes données 
    (2000-2020, 9 sites). En combinant les résultats, on obtient une **vue 
    complète** de la spécificité climatique de NOHEDES et de son évolution.
    """)

# ─────────────────────────────────────────────
# TAB 6 — RÉFÉRENCES
# ─────────────────────────────────────────────
with tab6:
    st.markdown("# 📚 Références")
    
    # ═══════════════════════════════════════════════
    # SECTION 1 — Contexte
    # ═══════════════════════════════════════════════
    st.markdown("## 🌸 1. Contexte")
    
    st.markdown("""
    **Delphinium montanum** est une **plante alpine endémique** des Pyrénées orientales 
    (France + Espagne), poussant entre **1600 et 2400 m**. Elle est classée comme 
    espèce en danger, avec seulement une douzaine de populations connues.
    
    🔗 [Fiche espèce — Floralab](https://www.floralab.eu/dauphinelle/)
    """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 2 — Le problème
    # ═══════════════════════════════════════════════
    st.markdown("## ⚠️ 2. Le problème observé")
    
    st.markdown("""
    À **NOHEDES**, aucune plante ne fleurit **depuis 2011** (Salvado et al., 2022). 
    La population est classée comme **la plus à risque d'extinction** de toutes les 
    localités de Delphinium montanum.
    
    Les plantes existent toujours (rosette, feuilles), mais **ne produisent plus de 
    tige florale ni de fleurs**, donc pas de graines et pas de renouvellement.
    
    🔗 [Article Salvado et al. 2022 — Wiley](https://onlinelibrary.wiley.com/doi/10.1002/ece3.8711)
    """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 3 — Seuils dans la littérature
    # ═══════════════════════════════════════════════
    st.markdown("## 📊 3. Seuils dans la littérature")
    
    st.markdown("""
    | Contexte | Type de mesure | Valeur | Article |
    |----------|----------------|--------|---------|
    | Delphinium horticoles | Sol < 7°C pendant 6-10 semaines | 7°C | [Alibaba Plant Care](https://lifetips.alibaba.com/plant-care/delphinium-flower-bloom-time) |
    | Alpine Andes (2475 m) | Air moyen annuel | 8.7°C | [Bruzzoni et al. 2021](https://www.mdpi.com/2223-7747/10/3/461) |
    | Prairie alpine Tibet | Air moyen annuel | 1.7°C | [Yang et al. 2020](https://www.frontiersin.org/articles/10.3389/fpls.2020.534703/full) |
    | Alpes autrichiennes | Impact perte de neige | Vernalisation compromise | [Wipf et al. 2009](https://link.springer.com/article/10.1007/s10584-008-9497-7) |
    
    ⚠️ **Il n'existe pas de seuil universel** — chaque espèce a son propre besoin.
    """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════
    # SECTION 4 — Références bibliographiques
    # ═══════════════════════════════════════════════
    st.markdown("## 📚 4. Références bibliographiques")
    
    st.markdown("""
    - **Salvado, J. et al. (2022).** *Little hope for the polyploid endemic Pyrenean 
      Larkspur (Delphinium montanum).* Ecology and Evolution, 12(3), e8711.  
      🔗 [Voir l'article](https://onlinelibrary.wiley.com/doi/10.1002/ece3.8711)
    
    - **Simon, J. et al. (2001).** *Conservation biology of the Pyrenean larkspur 
      (Delphinium montanum).* Biological Conservation, 98, 305-314.  
      🔗 [Voir l'article](https://www.sciencedirect.com/science/article/abs/pii/S0006320700001695)
    
    - **Yang, Z. et al. (2020).** *Responses of Plant Reproductive Phenology to 
      Winter-Biased Warming in an Alpine Meadow.* Frontiers in Plant Science.  
      🔗 [Voir l'article](https://www.frontiersin.org/articles/10.3389/fpls.2020.534703/full)
    
    - **Bruzzoni, R. et al. (2021).** *Flowering Phenology Adjustment in a South 
      American Alpine Species.* Plants, 10(3), 461.  
      🔗 [Voir l'article](https://www.mdpi.com/2223-7747/10/3/461)
    
    - **Broennimann, O. et al. (2021).** *Distance to native climatic niche margins 
      explains establishment success of alien mammals.* Nature Communications, 12, 2353.  
      🔗 [Voir l'article](https://www.nature.com/articles/s41467-021-22693-0)
    
    - **Wipf, S. et al. (2009).** *Winter climate change in alpine tundra: plant 
      responses to changes in snow depth and snowmelt timing.* Climatic Change, 94, 105-121.  
      🔗 [Voir l'article](https://link.springer.com/article/10.1007/s10584-008-9497-7)
    """)

# Footer
st.markdown("---")
st.caption("🌸 Application NOHEDES — Analyse de l'atypicité climatique | Stage D2ClimAFLo-Pyr")
