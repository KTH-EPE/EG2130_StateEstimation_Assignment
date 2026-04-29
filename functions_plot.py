import matplotlib.pyplot as plt
import pandapower.plotting as plot
import plotly.graph_objects as go
import json
import numpy as np

############################## Plotting functions ##############################
def box_plot(err_list, cases_names=None, title=None, ylabel=None):
    """ Box plot for error distributions across different cases 
    err_list[i]: list of error arrays for case i
    cases_names[i]: name of case i
    title[i]: title for subplot i
    ylabel[i]: y-axis label for subplot i
    """

    # Initialization
    n_plots = len(err_list)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    if cases_names is None:
        cases_names = [''] * n_plots
    if title is None:
        title = [''] * n_plots
    if ylabel is None:
        ylabel = [''] * n_plots

    colors = plt.cm.tab10(range(len(cases_names)))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Plot each error distribution
    for i in range(n_plots):
        bp = axes[i].boxplot(err_list[i], tick_labels=cases_names,
                             patch_artist=True, showmeans=True)

        for box, c in zip(bp['boxes'], colors):
            box.set_facecolor(c)

        axes[i].set_ylabel(ylabel[i], fontsize=11)
        axes[i].set_title(title[i], fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    return None

def grid_plot(net):
    """ Plot the grid with bus ids """
    sizes = plot.get_collection_sizes(net)

    # Create collections for buses, lines and transformers
    bc = plot.create_bus_collection(net, net.bus.index, color="red", size=sizes['bus'], zorder=2)
    lc = plot.create_line_collection(net, net.line.index, color="black", zorder=1)
    tc = plot.create_trafo_collection(net, net.trafo.index, color="black", zorder=1)

    # Add bus indices
    buses = net.bus.index.tolist() 
    coords = []
    
    for bus_idx in buses:        
        geo_str = net.bus.loc[bus_idx, 'geo']
        geo_data = json.loads(geo_str) 
        x, y = geo_data['coordinates']
        offset = 0.1
        coords.append((x + offset, y + offset))
    bic = plot.create_annotation_collection(
        size=sizes['bus'] * 1.5, 
        texts=np.char.mod('%d', buses), 
        coords=coords, 
        zorder=10, 
        color="black")
    plot.draw_collections([lc, tc, bc, bic], figsize=(20, 15)) 
    
    return buses

def err_plot(residuals, meas_errors, res_label="Residuals", err_label="Measurement Errors", title="Comparison of Residuals and Measurement Errors"):
    """ Bar plot comparison of vectors x_values and y_values
    default: x_values = residuals, y_values = measurement errors """

    # Initialization
    n = len(residuals)
    ids = list(range(1, n+1))  
    step = max(1, n // 20); tickvals = ids[::step]
    ticktext = [str(i) for i in tickvals]
    fig = go.Figure()

    # Bars for x_values (default = residuals)
    fig.add_trace(go.Bar(x=ids, y=residuals,
        name=res_label, marker_color='blue', opacity=0.7,
        hovertemplate='ID %{x}<br>'+res_label+': %{y:.4f}<extra></extra>'))

    # Bars for y_values (default = measurement errors)
    fig.add_trace(go.Bar(x=ids, y=meas_errors,
        name=err_label, marker_color='red', opacity=0.5,
        hovertemplate='ID %{x}<br>'+err_label+': %{y:.4f}<extra></extra>'))

    # Figure layout
    fig.update_layout(title=title,
        xaxis=dict(title='Measurement ID', tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title="Error"), barmode='overlay',bargap=0.1)

    fig.show()
    return None