import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict, Counter

def plot_cities(df: pd.DataFrame, airway_df: pd.DataFrame, railway_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 5))
    

    airway_cities = set(airway_df['city'].unique())
    railway_cities = set(railway_df['city'].unique())
    
    # Categorize each city
    def get_category(city):
        in_airway = city in airway_cities
        in_railway = city in railway_cities
        
        if in_airway and in_railway:
            return 'both'
        elif in_railway:
            return 'railway'
        elif in_airway:
            return 'airway'
        else:
            return 'none'
    
    df['transport_type'] = df["city"].apply(get_category)
    
    # Plot each category with different colors
    colors = {'none': 'steelblue', 'railway': 'green', 'airway': 'red', 'both': 'purple'}
    labels = {'none': 'Road only', 'railway': 'Road+Railway', 'airway': 'Road+Airway', 'both': 'All modes'}
    
    for category in ['none', 'railway', 'airway', 'both']:
        mask = df['transport_type'] == category
        ax.scatter(df[mask]['lng'], df[mask]['lat'], s=10, alpha=0.5, 
                  c=colors[category], label=labels[category])
    
    ax.set_xlim(65, 100)
    ax.set_ylim(5, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('analysis/cities_graph.png', dpi=200)
    plt.show()

def visualize_network(df, components_df, dist_df, weights, title='Network Analysis'):
    """
    Simple all-in-one network visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame - Cities data (city, lat, lng, population)
    components_df : pd.DataFrame - Floyd-Warshall components (b)
    dist_df : pd.DataFrame - Floyd-Warshall costs (a)
    weights : dict - Cost weights used
    title : str - Title for the visualization
    """
    
    cities = df['city'].values
    
    # Count direct vs multi-hop
    direct = 0
    multi_hop = 0
    mode_counts = {'road': 0, 'rail': 0, 'air': 0, 'mixed': 0}
    
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i >= j:
                continue
            
            comp = components_df.loc[src, dst]
            if isinstance(comp, dict):
                mode = comp['mode']
                if hasattr(mode, 'value'):
                    mode = mode.value
                
                if mode == 'mixed':
                    multi_hop += 1
                else:
                    direct += 1
                
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    total = direct + multi_hop
    
    # Calculate costs
    all_costs = []
    for src in cities:
        for dst in cities:
            if src != dst:
                cost = dist_df.loc[src, dst]
                if not np.isinf(cost):
                    all_costs.append(cost)
    
    # Print statistics
    print("\n" + "="*80)
    print(f"{title.upper()}")
    print("="*80)
    print(f"\nTotal cities: {len(cities)}")
    print(f"Total routes: {total}")
    print(f"Direct routes: {direct} ({direct/total*100:.1f}%)")
    print(f"Multi-hop routes: {multi_hop} ({multi_hop/total*100:.1f}%)")
    print(f"\nMode distribution:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {mode}: {count} ({count/total*100:.1f}%)")
    print(f"\nCost statistics:")
    print(f"  Average: {np.mean(all_costs):,.0f}")
    print(f"  Median: {np.median(all_costs):,.0f}")
    print(f"  Max: {np.max(all_costs):,.0f}")
    print("="*80 + "\n")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Geographic map
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax1.scatter(df['lng'], df['lat'], s=df['population']/5000, 
               alpha=0.5, c='steelblue', edgecolors='white', linewidth=0.5)
    ax1.set_xlim(65, 100)
    ax1.set_ylim(5, 40)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('City Network')
    ax1.grid(True, alpha=0.2)
    
    # 2. Mode distribution
    ax2 = fig.add_subplot(gs[0, 2])
    colors = {'road': '#1f77b4', 'rail': '#2ca02c', 'air': '#d62728', 'mixed': '#ff7f0e'}
    pie_colors = [colors.get(m, 'gray') for m in mode_counts.keys()]
    ax2.pie(mode_counts.values(), labels=mode_counts.keys(), autopct='%1.1f%%',
           colors=pie_colors, startangle=90)
    ax2.set_title('Mode Distribution')
    
    # 3. Cost distribution
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(all_costs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(np.mean(all_costs), color='red', linestyle='--', label='Mean')
    ax3.axvline(np.median(all_costs), color='orange', linestyle='--', label='Median')
    ax3.set_xlabel('Cost')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Cost Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    filename = title.lower().replace(' ', '_')
    plt.savefig(f'analysis/{filename}.png', dpi=150, bbox_inches='tight')
    plt.show()




def analyze_and_visualize_fc_network(df, fc_df, assignment_df, components_df):
    """
    Complete analysis and visualization of FC network in one function.
    
    Parameters:
    -----------
    df : pd.DataFrame - All cities (city, lat, lng, population)
    fc_df : pd.DataFrame - FC cities (city)
    assignment_df : pd.DataFrame - Result from assign_fcs() (city, assigned_fc, cost)
    components_df : pd.DataFrame - Floyd-Warshall components (b) for mode info
    
    Creates:
    --------
    - Comprehensive visualization (saved as PNG)
    - Prints statistics
    - Returns summary DataFrame
    """
    
    # Merge with city data
    full_data = assignment_df.merge(df[['city', 'lat', 'lng', 'population']], on='city')
    
    # Get mode information
    modes = []
    for _, row in assignment_df.iterrows():
        fc = row['fc']
        city = row['city']
        comp = components_df.loc[fc, city]
        if isinstance(comp, dict):
            try:
                mode = comp['mode']
            except:
                print(comp)
                    
            if hasattr(mode, 'value'):
                mode = mode.value
        else:
            print(components_df[fc][city])
            #mode = 'unknown'
        modes.append(mode)
    
    full_data['mode'] = modes
    
    # Calculate statistics
    print("\n" + "="*80)
    print("FC NETWORK ANALYSIS")
    print("="*80)
    
    # Basic stats
    cities_per_fc = assignment_df.groupby('fc').size().sort_values(ascending=False)
    print(f"\nTotal FCs: {len(fc_df)}")
    print(f"Active FCs: {len(cities_per_fc)}")
    print(f"Total cities: {len(assignment_df)}")
    print(f"Avg cities per FC: {cities_per_fc.mean():.1f}")
    
    # Cost stats
    print(f"\nCost Statistics:")
    print(f"  Average: {assignment_df['cost'].mean():,.0f}")
    print(f"  Median: {assignment_df['cost'].median():,.0f}")
    print(f"  Max: {assignment_df['cost'].max():,.0f}")
    
    # Top FCs
    print(f"\nTop 10 FCs by cities served:")
    print(cities_per_fc.head(10).to_string())
    
    # Mode distribution
    mode_counts = full_data['mode'].value_counts()
    print(f"\nMode Distribution:")
    print(mode_counts.to_string())
    
    # Population coverage
    pop_per_fc = full_data.groupby('fc')['population'].sum().sort_values(ascending=False)
    print(f"\nTop 10 FCs by population served:")
    for fc in pop_per_fc.head(10).index:
        print(f"  {fc}: {pop_per_fc[fc]/1e6:.2f}M people, {cities_per_fc[fc]} cities")
    
    print("="*80 + "\n")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Map with FC service areas
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_fc_map(ax1, df, fc_df, full_data)
    
    # 2. Cities per FC
    ax2 = fig.add_subplot(gs[0, 2])
    cities_per_fc.head(15).plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_xlabel('Cities Served')
    ax2.set_title('Top 15 FCs by Coverage')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Cost distribution
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(assignment_df['cost'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(assignment_df['cost'].mean(), color='red', linestyle='--', label='Mean')
    ax3.axvline(assignment_df['cost'].median(), color='orange', linestyle='--', label='Median')
    ax3.set_xlabel('Delivery Cost')
    ax3.set_ylabel('Number of Cities')
    ax3.set_title('Cost Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mode distribution pie
    ax4 = fig.add_subplot(gs[2, 0])
    colors = {'road': '#1f77b4', 'rail': '#2ca02c', 'air': '#d62728', 'mixed': '#ff7f0e'}
    pie_colors = [colors.get(m, 'gray') for m in mode_counts.index]
    ax4.pie(mode_counts.values, labels=mode_counts.index, autopct='%1.1f%%', 
            colors=pie_colors, startangle=90)
    ax4.set_title('Transport Mode Distribution')
    
    # 5. Population vs Cities
    ax5 = fig.add_subplot(gs[2, 1])
    fc_stats = pd.DataFrame({
        'cities': cities_per_fc,
        'population': pop_per_fc / 1e6
    }).head(15)
    x = np.arange(len(fc_stats))
    ax5.bar(x - 0.2, fc_stats['cities'], 0.4, label='Cities', color='steelblue')
    ax5_twin = ax5.twinx()
    ax5_twin.bar(x + 0.2, fc_stats['population'], 0.4, label='Pop (M)', color='orange')
    ax5.set_xticks(x)
    ax5.set_xticklabels(fc_stats.index, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Cities', color='steelblue')
    ax5_twin.set_ylabel('Population (M)', color='orange')
    ax5.set_title('Load Distribution (Top 15 FCs)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Stats box
    ax6 = fig.add_subplot(gs[2, 2])
    stats_text = f"""
    SUMMARY STATISTICS
    
    Total FCs: {len(fc_df)}
    Active FCs: {len(cities_per_fc)}
    Total Cities: {len(assignment_df)}
    
    Avg Cost: {assignment_df['cost'].mean():,.0f}
    Median Cost: {assignment_df['cost'].median():,.0f}
    
    Avg Cities/FC: {cities_per_fc.mean():.1f}
    Max Cities/FC: {cities_per_fc.max()}
    
    Total Population: {full_data['population'].sum()/1e6:.1f}M
    """
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.axis('off')
    
    plt.suptitle('Fulfillment Center Network Analysis', fontsize=16, fontweight='bold')
    plt.savefig('analysis/fc_network_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Return summary dataframe
    summary = pd.DataFrame({
        'fc': cities_per_fc.index,
        'cities_served': cities_per_fc.values,
        'population_served': [pop_per_fc[fc] for fc in cities_per_fc.index],
        'avg_cost': [full_data[full_data['fc']==fc]['cost'].mean() for fc in cities_per_fc.index]
    })
    
    return summary

def plot_fc_map(ax, df, fc_df, full_data):
    """Helper to plot the geographic FC network map with emphasized air routes"""
    
    fc_cities = set(fc_df['city'].values)
    city_to_coords = dict(zip(df['city'], zip(df['lng'], df['lat'])))
    
    # Get ALL FCs for coloring
    cities_per_fc = full_data.groupby('fc').size().sort_values(ascending=False)
    all_fcs = cities_per_fc.index.tolist()
    
    # Color mapping
    n_fcs = len(all_fcs)
    if n_fcs <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_fcs))
    else:
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, n_fcs))
    fc_colors = dict(zip(all_fcs, colors))
    
    # Draw connections
    for fc in all_fcs:
        fc_data = full_data[full_data['fc'] == fc]
        fc_coord = city_to_coords.get(fc)
        
        if not fc_coord:
            continue
        
        for _, row in fc_data.iterrows():
            if row['city'] == fc:
                continue
            city_coord = city_to_coords.get(row['city'])
            if not city_coord:
                continue
            
            # --- CUSTOM LOGIC FOR AIR ROUTES ---
            if row['mode'] == 'air':
                # Draw a curved dark line for Air
                # rad=0.3 creates the arc. Use a positive value for one-way curvature.
                style = "arc3,rad=0.3"
                arrow = patches.FancyArrowPatch(
                    fc_coord, city_coord,
                    connectionstyle=style,
                    color='black',          # Dark color for visibility
                    linestyle='--',         # Dotted/Dashed
                    linewidth=1.2,          # Slightly thicker
                    alpha=0.6,              # Higher opacity than ground
                    arrowstyle='-',         # No actual arrowhead needed
                    zorder=4                # Ensure it's above other lines
                )
                ax.add_patch(arrow)
            else:
                # Draw standard straight line for Road/Rail/Mixed
                ax.plot([fc_coord[0], city_coord[0]], [fc_coord[1], city_coord[1]],
                       color=fc_colors[fc], 
                       alpha=0.1,           # Very faint for ground routes
                       linewidth=0.5, 
                       zorder=1)
    
    # Plot cities
    for fc in all_fcs:
        fc_data = full_data[full_data['fc'] == fc]
        cities_in_fc = fc_data['city'].values
        city_df = df[df['city'].isin(cities_in_fc)]
        
        ax.scatter(city_df['lng'], city_df['lat'], 
                  s=city_df['population']/20000, # Adjusted scale for clarity
                  alpha=0.4, 
                  c=[fc_colors[fc]], 
                  zorder=2,
                  edgecolors='white',
                  linewidth=0.2)
    
    # Plot FCs as prominent markers
    fc_plot_data = df[df['city'].isin(fc_cities)]
    ax.scatter(fc_plot_data['lng'], fc_plot_data['lat'], s=120, c='red', marker='*',
              zorder=5, edgecolors='black', linewidth=0.8, label='Fulfillment Centers')
    
    # Formatting
    ax.set_xlim(65, 100)
    ax.set_ylim(5, 40)
    ax.set_facecolor('#f8f9fa') # Light grey background for better contrast
    ax.set_title('FC Network: Ground vs. Air Distribution\n(Dashed arcs indicate Air routes)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.1, linestyle=':')