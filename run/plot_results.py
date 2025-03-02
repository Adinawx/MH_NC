import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import os

def set_tick_rotation(rotation_angle=45):
    plt.xticks(rotation=rotation_angle)
    # plt.yticks(rotation=rotation_angle)

def print_for_paper(cfg, dfs, labels, er_rates, node_value=None, filename=None):
    """
    Plot metrics for any number of DataFrames with labels, generating separate figures for each metric.
    Also creates a combined plot for Channel Usage Rate for all n != -1.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames to plot metrics from.
    labels (list of str): List of labels corresponding to each DataFrame.
    er_rates (list of float): Error rates for calculating capacity.
    node_value (int): Node value to filter data. If None, include all nodes.
    filename (str): Directory to save figures.
    """

    # Set Figure Features: ####################################################
    # colors:
    # custom_colors = ['#00A2E8', '#D10056', '#C4A000', '#000000', 
    #                 '#1D3557', '#006400', '#FF6347', 
    #                 '#20B2AA']
    fig_size = (10, 8)
    markersize = 25
    line_width = 5
    plt.rcParams["mathtext.fontset"] = "cm"

    plt.rcParams.update({
        'font.size': 24,          # Base font size
        'axes.titlesize': 24,     # Title font size
        'axes.labelsize': 24,     # Axes label font size
        'xtick.labelsize': 24,    # X-tick label font size
        'ytick.labelsize': 24,    # Y-tick label font size
        'legend.fontsize': 21,    # Legend font size 22.8
        'figure.titlesize': 24,    # Figure title font size
        })

    # Set the default line width for all plots
    plt.rcParams.update({
        'lines.linewidth': line_width # Default line width for all lines
    })

    custom_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        # '#e377c2',  # Pink
        # '#7f7f7f',  # Gray
        # '#bcbd22',  # Olive
        '#17becf',   # Teal
        '#006400', 
        '#FF6347', 
        '#20B2AA'
    ]

    markers = [
    ',',  # Pixel marker
    '*',  # Star
    'o',  # Circle marker
    'D',  # Diamond 
    'h',  # Triangle up
    '<',  # Triangle left
    '>',  # Triangle right
    '1',  # Tri-down
    '2',  # Tri-up
    '3',  # Tri-left
    '4',  # Tri-right
    's',  # Square
    'p',  # Pentagon
    '.',  # Point
    'h',  # Hexagon1
    'H',  # Hexagon2
    '+',  # Plus
    'x',  # Cross
    'v',  # Triangle down
    'd',  # Thin diamond
    '|',  # Vertical line
    '_',  # Horizontal line
    ]

    # X axis:
    all_eps = dfs[0]['Eps'].unique()
    # Capcity:
    eps_bn_fixed = max(er_rates)
    eps_bn_changed = np.array([max(eps_bn_fixed, eps) for eps in all_eps])
    capacity = 1 - eps_bn_changed
    # SR-ARQ Results path:
    srArq_path = os.path.join(cfg.param.project_folder, 'Results_SR_ARQ')
    ###########################################################################

    # Calculate mean values for each metric ###################################
    dfs_mean = [df.groupby(['Node', 'Eps']).mean().reset_index() for df in dfs]
    ###########################################################################

    # Plot Channel Usage Rate for all nodes on the same figure #######################################################
    node_value = dfs_mean[0]['Node'].unique() # All nodes
    N = max(node_value) + 1 

    plt.figure(figsize=fig_size)

    for idx, n in enumerate(node_value):
        
        dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node

        for df_node, label in zip(dfs_mean_node, labels): 
               
            # for any label != bs_empty plot only once:
            if label != "BS-EMPTY": # and n != 1:
                continue

            # color = custom_colors[idx % len(custom_colors)] # Cycle through colors
            if idx < 5:
             color = custom_colors[idx] # Cycle through colors
            elif idx == 5:
                color = '#7f7f7f'

            # Correct paper labels #####
            if label == "BS-EMPTY":
                plt_label = 'BS'
                markersize_ = markersize + 10 # Cause the * is very small
            # elif label == "AC-FEC":
            #     plt_label = 'NET-FEC'
            #     markersize_ = markersize 
            # elif label == "MIXALL":
            #     plt_label = 'Baseline'
            #     markersize_ = markersize 
            # else:
            #     plt_label = label
            #     markersize_ = markersize 
            ############################

            label_ = f'{plt_label} - Node {n-1}' if n != -1 else f"{plt_label} - End to End"

            plt.plot(df_node['Eps'], df_node['Channel Usage Rate'], 
                    label=label_, 
                    linestyle='-', alpha=0.7 if n != -1 else 0.85, 
                    linewidth=line_width-1 if n != -1 else line_width, 
                    color = color if n != -1 else custom_colors[-1],
                    marker=markers[idx+1],
                    markerfacecolor='none' if n != -1 else custom_colors[-1],
                    markeredgecolor=color,
                    markeredgewidth=3 if n != -1 else 1.5,
                    markersize=markersize_-20 if n !=-1 else markersize_,
                    zorder=10 if n == -1 else 5)


    # # Add SR-ARQ as 1s: #################
    others = np.ones(len(all_eps))
    plt.plot(all_eps, others, 
                label='Baseline, Local-ACRLNC, SR-ARQ', 
                linestyle='-', alpha=0.7, 
                # linewidth=3, 
                color='#8c564b', marker='p', markersize=markersize)
    # ####################################

    plt.xlabel(r"$\epsilon_2$", fontsize=46)
    # plt.ylabel('Channel Usage Rate', fontsize=14)
    plt.ylim(0, 1.02)
    # plt.xlim(min(all_eps), max(all_eps))
    plt.xlim(0.2, 0.6)
    plt.xticks(np.arange(0.2, 0.7, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(framealpha=0.0)
    # plt.tick_params(axis='both', which='major', labelsize=14)  # Tick labels
    set_tick_rotation()
    
    plt.tight_layout() 
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(f'{filename}/Channel_Usage_Rate_All_vs_eps.png', dpi=300)
    ######################################################################################################################

    # Plot rate metrics for last node  - One Figure #########################################################################
    n = -1
    dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node    
    
    all_lines = []
    all_labels = []

    plt.figure(figsize=fig_size)

    # All Delivery Rate: #############################################
    column = 'Delivery Rate'
    for df_node, label in zip(dfs_mean_node, labels):
        
        # Correct paper labels and colors ####
        if label == "BS-EMPTY":
            plt_label = 'BS- $R_{del}$'
            color = custom_colors[-1]
            marker = markers[1]
            markersize_ = markersize + 10 # Cause the * is very small
        elif label == "AC-FEC":
            plt_label = 'NET-FEC'
            color = custom_colors[-2]
            marker = markers[2]
            markersize_ = markersize
        elif label == "MIXALL":
            plt_label = 'Baseline'
            color = custom_colors[-3]
            marker = markers[3]
            markersize_ = markersize
        else:
            plt_label = label
            color = custom_colors[-4]
            marker = markers[4]
            markersize_ = markersize
        ########################################

        line, = plt.plot(df_node['Eps'], df_node[column], 
                    label=f'{plt_label}', 
                    linestyle='-', alpha=0.7, 
                #  linewidth=3, 
                    color=color, marker=marker, markersize=markersize_, zorder=1)
        all_lines.append(line)
        all_labels.append(f'{plt_label}')
        #####################################################################

    # BS Normalized Goodput: #############################################
    column = 'Normalized Goodput'
    for df_node, label in zip(dfs_mean_node, labels):
        
        # Correct paper labels and colors ####
        if label == "BS-EMPTY":
            plt_label = 'BS - $\eta$'
            color = custom_colors[-1]
            marker = '^'
            markersize_ = markersize
        else:
            continue
        ########################################

        bs_g, = plt.plot(df_node['Eps'], df_node[column], 
                label=f'{plt_label}', 
                linestyle='--', alpha=0.7, 
                #  linewidth=3, 
                color=color, marker=marker, markersize=markersize_, 
                # markeredgewidth=3,
                zorder=2)
        all_lines.append(bs_g)
        all_labels.append(f'{plt_label}')
    #####################################################################
    
    # Add SR-ARQ Results: ################################################
    srArq_filename = f'{srArq_path}/node_{N}/srARQ_tau.txt'
    srArq_data = np.loadtxt(srArq_filename, delimiter=',')
    srArq_eps = srArq_data[:, 0]
    srArq_values = srArq_data[:, 1]
    
    line, = plt.plot(srArq_eps, srArq_values, 
                label='SR-ARQ',
                linestyle='-', alpha=0.7, 
                # linewidth=3,
                color=custom_colors[-5], marker=markers[5], markersize=markersize)
    all_lines.append(line)
    all_labels.append('SR-ARQ')
    #######################################################################

    # Add capacity line for Delivery Rate
    # if n == -1 and column == 'Delivery Rate':
    plt.plot(all_eps, capacity, label='Capacity', linestyle='--', color='black', linewidth=2)
    all_lines.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2))
    all_labels.append('Capacity')

    plt.xlabel(r"$\epsilon_2$", fontsize=46)
    plt.ylim(0, 1.01)
    # plt.xlim(min(all_eps), max(all_eps))
    plt.xlim(min(srArq_eps), max(srArq_eps))
    plt.xticks(np.arange(0.2, 0.7, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Reorder legend entries: BS-G is first
    line_first = bs_g
    label_first = 'BS - $\eta$'
    all_lines.remove(bs_g)
    all_labels.remove('BS - $\eta$')
    all_lines.insert(0, line_first)
    all_labels.insert(0, label_first)

    plt.legend(all_lines, all_labels, handlelength=3, facecolor='white', framealpha=1, loc='lower left')
     
    # plt.tick_params(axis='both', which='major', labelsize=12)  # Tick labels
    set_tick_rotation()
    plt.tight_layout()

    # Save figure for the current metric
    if filename:
        plt.savefig(f'{filename}/Node_{n}_Data_Rates_vs_eps.png', dpi=300)
    ######################################################################################################################
  
    # Plot rate metrics for last node - Two Figures ####################################################################################
    # n = -1
    # dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node
    # metrics_to_plot = ['Normalized Goodput', 'Delivery Rate']
    
    # for column in metrics_to_plot:
    #     # Create a separate figure for each metric
    #     plt.figure(figsize=fig_size)

    #     for df_node, label in zip(dfs_mean_node, labels):
            
    #         # Correct paper labels and colors ####
    #         if label == "BS-EMPTY":
    #             plt_label = 'BS'
    #             color = custom_colors[-1]
    #             marker = markers[1]
    #             markersize_ = markersize + 10 # Cause the * is very small
    #         elif label == "AC-FEC":
    #             plt_label = 'NET-FEC'
    #             color = custom_colors[-2]
    #             marker = markers[2]
    #             markersize_ = markersize
    #         elif label == "MIXALL":
    #             plt_label = 'Baseline'
    #             color = custom_colors[-3]
    #             marker = markers[3]
    #             markersize_ = markersize
    #         else:
    #             plt_label = label
    #             color = custom_colors[-4]
    #             marker = markers[4]
    #             markersize_ = markersize
    #         ########################################

    #         plt.plot(df_node['Eps'], df_node[column], 
    #                  label=f'{plt_label}', 
    #                  linestyle='-', alpha=0.7, 
    #                 #  linewidth=3, 
    #                  color=color, marker=marker, markersize=markersize_)

    #     # Add SR-ARQ Results: ################################################
    #     srArq_filename = f'{srArq_path}/node_{N}/srARQ_tau.txt'
    #     srArq_data = np.loadtxt(srArq_filename, delimiter=',')
    #     srArq_eps = srArq_data[:, 0]
    #     srArq_values = srArq_data[:, 1]
        
    #     plt.plot(srArq_eps, srArq_values, 
    #                 label='SR-ARQ',
    #                 linestyle='-', alpha=0.7, 
    #                 # linewidth=3,
    #                 color=custom_colors[-5], marker=markers[5], markersize=markersize)
    #     #######################################################################

    #     # Add capacity line for Delivery Rate
    #     if n == -1 and column == 'Delivery Rate':
    #         plt.plot(all_eps, capacity, label='Capacity', linestyle='--', color='black', linewidth=2)

    #     plt.xlabel(r"$\epsilon_2$", fontsize=46)
    #     plt.ylim(0, 1.01)
    #     # plt.xlim(min(all_eps), max(all_eps))
    #     plt.xlim(min(srArq_eps), max(srArq_eps))
    #     plt.xticks(np.arange(0.2, 0.7, 0.1))
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend()
    #     # plt.tick_params(axis='both', which='major', labelsize=12)  # Tick labels
    #     plt.tight_layout()
    #     set_tick_rotation()
    #     plt.tight_layout() 

    #     # Save figure for the current metric
    #     if filename:
    #         plt.savefig(f'{filename}/Node_{n}_{column.replace(" ", "_")}_vs_eps.png', dpi=300)
    #     ######################################################################################################################

    # Plot dealy metrics for last nodes ####################################################################################
    node_value = dfs_mean[0]['Node'].unique() # All nodes
    N = max(node_value)
    metrics_to_plot = ['Mean Delay', 'Max Delay']
    

    for metric in metrics_to_plot:
        plt.figure(figsize=fig_size)

        for idx, n in enumerate(node_value):
            
            if n != -1:
                continue

            dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node
                        
            for df_node, label in zip(dfs_mean_node, labels):                            

                # Correct paper labels #####
                if label == "BS-EMPTY":
                    plt_label = 'BS'
                    color = custom_colors[-1]
                    marker = markers[1]
                    markersize_ = markersize + 10 # Cause the * is very small
                elif label == "AC-FEC":
                    plt_label = 'NET-FEC'
                    color = custom_colors[-2]
                    marker = markers[2]
                    markersize_ = markersize
                elif label == "MIXALL":
                    plt_label = 'Baseline'
                    color = custom_colors[-3]
                    marker = markers[3]
                    markersize_ = markersize
                else:
                    plt_label = label
                    color = custom_colors[-4]
                    marker = markers[4]
                    markersize_ = markersize
                ##############################
                
                label_ = f'{plt_label} node {n}' if n != -1 else f'{plt_label}'

                plt.plot(df_node['Eps'], df_node[metric], 
                        label = label_,
                        linestyle='-', alpha=0.7, 
                        # linewidth=2.5, 
                        color=color, marker=marker,
                        # markerfacecolor='none',  
                        # markeredgecolor=color,
                        markersize=markersize_,           
                        markeredgewidth=1.5)        

            # Add SR-ARQ Results: ################################################
            if 'Mean' in metric:
                if n != -1:
                    srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmean.txt'
                else:
                    srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmean.txt'

            elif 'Max' in metric:
                if n != -1:
                    srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmax.txt'
                else:
                    srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmax.txt'
                        
            srArq_data = np.loadtxt(srArq_filename, delimiter=',')
            srArq_eps = srArq_data[:, 0]
            srArq_values = srArq_data[:, 1]
            
            plt.plot(srArq_eps, srArq_values, 
                        label = f'SR-ARQ node {n}' if n != -1 else 'SR-ARQ',
                        linestyle='-', alpha=0.7, 
                        # linewidth=2.5, 
                        color=custom_colors[-5], marker=markers[5],
                        # markerfacecolor='none', 
                        # markeredgecolor=custom_colors[-5],
                        markersize=markersize,           
                        markeredgewidth=1.5)       
            #######################################################################


        plt.xlabel(r"$\epsilon_2$", fontsize=46)#, fontsize=14)
        plt.xlim(min(srArq_eps), max(srArq_eps))
        plt.xticks(np.arange(0.2, 0.7, 0.1))

        # Set y lim:
        if 'Mean' in metric:
            plt.ylim(0, 3600)
        elif 'Max' in metric:
            plt.ylim(0, 3600)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        # plt.legend(loc='upper left')#, bbox_to_anchor=(0, -0.5))
        # plt.tick_params(axis='both', which='major', labelsize=14)  # Tick labels
        set_tick_rotation()
        plt.tight_layout() 

        # Save figure if filename is provided
        if filename:
            plt.savefig(f'{filename}/{metric}_All_vs_eps.png', dpi=300)
        ######################################################################################################################

    ### Semi Decoding ########################################################################
    # node_value = dfs_mean[0]['Node'].unique() # All nodes
    # N = max(node_value) + 1 
    # metrics_to_plot = ['Mean Delay', 'Max Delay']
    

    # for metric in metrics_to_plot:
    #     plt.figure(figsize=(10, 8))

    #     for idx, n in enumerate(node_value):
    #         dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node

    #         marker = markers[idx % len(markers)] # Cycle through markers
            
    #         for df_node, label in zip(dfs_mean_node, labels):                            

    #             # Correct paper labels #####
    #             if label == "BS-EMPTY":
    #                 plt_label = 'BS-ACRLNC'
    #                 color = custom_colors[-1]
    #             elif label == "AC-FEC":
    #                 plt_label = 'Local-ACRLNC'
    #                 color = custom_colors[-2]
    #             elif label == "MIXALL":
    #                 plt_label = 'Baseline'
    #                 color = custom_colors[-3]
    #             else:
    #                 plt_label = label
    #                 color = custom_colors[-4]
    #             ##############################

    #             plt.plot(df_node['Eps'], df_node[metric], 
    #                     label=f'{plt_label} - node {n-1 if n != -1 else "All"}', 
    #                     linestyle='-', alpha=0.7, 
    #                     linewidth=1.5, color=color, marker=marker)

    #         # Add SR-ARQ Results: ################################################
    #         if 'Mean' in metric:
    #             if n != -1:
    #                 srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmean.txt'
    #             else:
    #                 srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmean.txt'

    #         elif 'Max' in metric:
    #             if n != -1:
    #                 srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmax.txt'
    #             else:
    #                 srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmax.txt'
                        
    #         srArq_data = np.loadtxt(srArq_filename, delimiter=',')
    #         srArq_eps = srArq_data[:, 0]
    #         srArq_values = srArq_data[:, 1]
            
    #         plt.plot(srArq_eps, srArq_values, 
    #                     label=f'SR-ARQ - node {n-1 if n != -1 else "All"}',
    #                     linestyle='-', alpha=0.7, 
    #                     linewidth=1.5, color=custom_colors[-5], marker=marker)
    #         #######################################################################


    #     plt.xlabel(r"$\epsilon_2$", fontsize=46, fontsize=14)
    #     plt.xlim(min(srArq_eps), max(srArq_eps))
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(fontsize=12)
    #     plt.tight_layout()

    #     # Save figure if filename is provided
    #     if filename:
    #         plt.savefig(f'{filename}/{metric}_All_vs_eps.png', dpi=300)

def print_for_arxiv(cfg, dfs, labels, er_rates, node_value=None, filename=None):
    """
    Plot metrics for any number of DataFrames with labels, generating separate figures for each metric.
    Also creates a combined plot for Channel Usage Rate for all n != -1.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames to plot metrics from.
    labels (list of str): List of labels corresponding to each DataFrame.
    er_rates (list of float): Error rates for calculating capacity.
    node_value (int): Node value to filter data. If None, include all nodes.
    filename (str): Directory to save figures.
    """

    # Set Figure Features: ####################################################
    # colors:
    # custom_colors = ['#00A2E8', '#D10056', '#C4A000', '#000000', 
    #                 '#1D3557', '#006400', '#FF6347', 
    #                 '#20B2AA']
    fig_size = (10, 8)
    markersize = 25
    line_width = 5
    
    plt.rcParams["mathtext.fontset"] = "cm"


    plt.rcParams.update({
        'font.size': 24,          # Base font size
        'axes.titlesize': 24,     # Title font size
        'axes.labelsize': 24,     # Axes label font size
        'xtick.labelsize': 24,    # X-tick label font size
        'ytick.labelsize': 24,    # Y-tick label font size
        'legend.fontsize': 21,    # Legend font size 22.8
        'figure.titlesize': 24    # Figure title font size
    })

    # Set the default line width for all plots
    plt.rcParams.update({
        'lines.linewidth': line_width # Default line width for all lines
    })

    custom_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        # '#e377c2',  # Pink
        # '#7f7f7f',  # Gray
        # '#bcbd22',  # Olive
        '#17becf',   # Teal
        '#006400', 
        '#FF6347', 
        '#20B2AA'
    ]

    markers = [
    ',',  # Pixel marker
    '*',  # Star
    'o',  # Circle marker
    'D',  # Diamond 
    'h',  # Triangle up
    '<',  # Triangle left
    '>',  # Triangle right
    '1',  # Tri-down
    '2',  # Tri-up
    '3',  # Tri-left
    '4',  # Tri-right
    's',  # Square
    'p',  # Pentagon
    '.',  # Point
    'h',  # Hexagon1
    'H',  # Hexagon2
    '+',  # Plus
    'x',  # Cross
    'v',  # Triangle down
    'd',  # Thin diamond
    '|',  # Vertical line
    '_',  # Horizontal line
    ]

    # X axis:
    all_eps = dfs[0]['Eps'].unique()
    # Capcity:
    eps_bn_fixed = max(er_rates)
    eps_bn_changed = np.array([max(eps_bn_fixed, eps) for eps in all_eps])
    capacity = 1 - eps_bn_changed
    # SR-ARQ Results path:
    srArq_path = os.path.join(cfg.param.project_folder, 'Results_SR_ARQ')
    ###########################################################################

    # Calculate mean values for each metric ###################################
    dfs_mean = [df.groupby(['Node', 'Eps']).mean().reset_index() for df in dfs]
    ###########################################################################

    max_n = max(dfs_mean[0]['Node'].unique())
    
    
    for curr_n in range(1, max_n+1):
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))


        # Plot rate metrics for last node  - One Figure #########################################################################
        n = curr_n
        dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node    
        
        all_lines = []
        all_labels = []

        # plt.figure(figsize=fig_size)
        
        # All Delivery Rate: #############################################
        column = 'Delivery Rate Real'
        ax = axs[0]
        for df_node, label in zip(dfs_mean_node, labels):
            
            # Correct paper labels and colors ####
            if label == "BS-EMPTY":
                plt_label = 'BS- $R_{del}$'
                color = custom_colors[-1]
                marker = markers[1]
                markersize_ = markersize + 10 # Cause the * is very small
            elif label == "AC-FEC":
                plt_label = 'NET-FEC'
                color = custom_colors[-2]
                marker = markers[2]
                markersize_ = markersize
            elif label == "MIXALL":
                plt_label = 'Baseline'
                color = custom_colors[-3]
                marker = markers[3]
                markersize_ = markersize
            else:
                plt_label = label
                color = custom_colors[-4]
                marker = markers[4]
                markersize_ = markersize
            ########################################

            line, = ax.plot(df_node['Eps'], df_node[column], 
                        label=f'{plt_label}', 
                        linestyle='-', alpha=0.7, 
                    #  linewidth=3, 
                        color=color, marker=marker, markersize=markersize_, zorder=1)
            all_lines.append(line)
            all_labels.append(f'{plt_label}')
            #####################################################################

        # BS Normalized Goodput: #############################################
        column = 'Normalized Goodput Real'
        for df_node, label in zip(dfs_mean_node, labels):
            
            # Correct paper labels and colors ####
            if label == "BS-EMPTY":
                plt_label = 'BS - $\eta$'
                color = custom_colors[-1]
                marker = '^'
                markersize_ = markersize + 10 # Cause the * is very small
            else:
                continue
            ########################################

            bs_g, = ax.plot(df_node['Eps'], df_node[column], 
                    label=f'{plt_label}', 
                    linestyle='--', alpha=0.7, 
                    #  linewidth=3, 
                    color=color, marker=marker, markersize=markersize_, zorder=2)
            all_lines.append(bs_g)
            all_labels.append(f'{plt_label}')
        #####################################################################
        
        # Add SR-ARQ Results: ################################################
        srArq_filename = f'{srArq_path}/node_{curr_n+1}/srARQ_tau.txt'
        srArq_data = np.loadtxt(srArq_filename, delimiter=',')
        srArq_eps = srArq_data[:, 0]
        srArq_values = srArq_data[:, 1]
        
        line, = ax.plot(srArq_eps, srArq_values, 
                    label='SR-ARQ',
                    linestyle='-', alpha=0.7, 
                    # linewidth=3,
                    color=custom_colors[-5], marker=markers[5], markersize=markersize)
        all_lines.append(line)
        all_labels.append('SR-ARQ')
        #######################################################################

        # Add capacity line for Delivery Rate
        # if n == -1 and column == 'Delivery Rate':
        ax.plot(all_eps, capacity, label='Capacity', linestyle='--', color='black', linewidth=2)

        ax.set_xlabel(r"$\epsilon_2$", fontsize=46)
        ax.set_ylim(0, 1.01)
        # plt.xlim(min(all_eps), max(all_eps))
        ax.set_xlim(min(srArq_eps), max(srArq_eps))
        ax.set_xticks(np.arange(0.2, 0.7, 0.1))
        ax.grid(True, linestyle='--', alpha=0.7)

        # Reorder legend entries: BS-G is first
        line_first = bs_g
        label_first = 'BS - $\eta$'
        all_lines.remove(bs_g)
        all_labels.remove('BS - $\eta$')
        all_lines.insert(0, line_first)
        all_labels.insert(0, label_first)

        ax.legend(all_lines, all_labels, handlelength=3, facecolor='white', framealpha=1, loc='lower left')
        
        ax.set_title('$R_{del}$, $\eta$')
        # plt.tick_params(axis='both', which='major', labelsize=12)  # Tick labels
        # set_tick_rotation()
        # plt.xticks(rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax.tight_layout() 

        # Save figure for the current metric
        # if filename:
        #     ax.savefig(f'{filename}/Node_{n}_Data_Rates_vs_eps.png', dpi=300)
        ######################################################################################################################
    
 
        # Plot dealy metrics for last nodes ####################################################################################
        node_value = dfs_mean[0]['Node'].unique() # All nodes
        N = curr_n ###
        metrics_to_plot = ['Mean Real Delay', 'Max Real Delay']
        

        for metric in metrics_to_plot:
            # plt.figure(figsize=fig_size)
            ax = axs[1] if 'Mean' in metric else axs[2]

            for idx, n in enumerate(node_value):
                
                if n != N:
                    continue

                dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean] # Filter to only the current node
                            
                for df_node, label in zip(dfs_mean_node, labels):                            

                    # Correct paper labels #####
                    if label == "BS-EMPTY":
                        plt_label = 'BS'
                        color = custom_colors[-1]
                        marker = markers[1]
                        markersize_ = markersize + 10 # Cause the * is very small
                    elif label == "AC-FEC":
                        plt_label = 'NET-FEC'
                        color = custom_colors[-2]
                        marker = markers[2]
                        markersize_ = markersize
                    elif label == "MIXALL":
                        plt_label = 'Baseline'
                        color = custom_colors[-3]
                        marker = markers[3]
                        markersize_ = markersize
                    else:
                        plt_label = label
                        color = custom_colors[-4]
                        marker = markers[4]
                        markersize_ = markersize
                    ##############################
                    
                    label_ = f'{plt_label}' #if n != N else f'{plt_label} - Destination'

                    ax.plot(df_node['Eps'], df_node[metric], 
                            label = label_,
                            linestyle='-', alpha=0.7, 
                            # linewidth=2.5, 
                            color=color, marker=marker,
                            # markerfacecolor='none',  
                            # markeredgecolor=color,
                            markersize=markersize_,           
                            markeredgewidth=1.5)        

                # Add SR-ARQ Results: ################################################
                if 'Mean' in metric:
                    if n != -1:
                        srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmean.txt'
                    else:
                        srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmean.txt'

                elif 'Max' in metric:
                    if n != -1:
                        srArq_filename = f'{srArq_path}/node_{n+1}/srARQ_dmax.txt'
                    else:
                        srArq_filename = f'{srArq_path}/node_{N}/srARQ_dmax.txt'
                            
                srArq_data = np.loadtxt(srArq_filename, delimiter=',')
                srArq_eps = srArq_data[:, 0]
                srArq_values = srArq_data[:, 1]
                
                ax.plot(srArq_eps, srArq_values, 
                            label = f'SR-ARQ', #if n != N else 'SR-ARQ - Destination',
                            linestyle='-', alpha=0.7, 
                            # linewidth=2.5, 
                            color=custom_colors[-5], marker=markers[5],
                            # markerfacecolor='none', 
                            # markeredgecolor=custom_colors[-5],
                            markersize=markersize,           
                            markeredgewidth=1.5)       
                #######################################################################


            ax.set_xlabel(r"$\epsilon_2$", fontsize=46)#, fontsize=14)
            ax.set_xlim(min(srArq_eps), max(srArq_eps))
            ax.set_xticks(np.arange(0.2, 0.7, 0.1))
            # set ylim:
            if 'Mean' in metric:
                ax.set_title('Mean Delay [Slots]')
                if curr_n == 1:
                    ax.set_ylim(0, 100)                    
                else:   
                    ax.set_ylim(0, 3600)
            elif 'Max' in metric:
                ax.set_title('Max Delay [Slots]')
                if curr_n == 1:
                    ax.set_ylim(0, 100)
                else:   
                    ax.set_ylim(0, 3600)

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            # plt.legend(loc='upper left')#, bbox_to_anchor=(0, -0.5))
            # plt.tick_params(axis='both', which='major', labelsize=14)  # Tick labels
            # set_tick_rotation()
            # ax.tight_layout() 
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            # Save figure if filename is provided
            # if filename:
            #     plt.savefig(f'{filename}/{metric}_{curr_n}_vs_eps.png', dpi=300)

        # Adjust the layout and save the figure
        # for ax in axs:
        #     ax.legend(loc="best")

        plt.tight_layout()
        if filename:
            plt.savefig(f'{filename}/Node_{curr_n}_Combined_Metrics_vs_eps.png', dpi=300)

            ######################################################################################################################
