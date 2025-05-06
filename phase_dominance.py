"""
Phase Dominance Analysis Module

This module provides functions for analyzing and visualizing the dominance of components
in multi-dimensional phase spaces. It supports both linear and piecewise linear fitting methods
to characterize phase separation behavior.

Typical usage:
    1. Create 2D visualizations of phase spaces using plot_shaded_grid
    2. Calculate dominance using dominance_sweep or dominance_sweep_piecewiselinear
    3. Visualize dominance trends across different slices of the phase space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
import pandas as pd


def plot_shaded_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    avg_col: str,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    N: int,
    minDroplets: int,
    cmap: Any,
    vlims: Optional[Tuple[float, float]],
    ax: plt.Axes
) -> plt.Axes:
    """
    Plot a grid shaded by the average value of a specified column within each square.
    
    Parameters:
    ----------
    df : DataFrame
        DataFrame containing the data to visualize.
    x_col : str
        Column name for x-values.
    y_col : str
        Column name for y-values.
    avg_col : str
        Column name for calculating averages (determines the color).
    x_lim : Tuple[float, float]
        (min, max) for x-axis limits.
    y_lim : Tuple[float, float]
        (min, max) for y-axis limits.
    N : int
        Number of squares per grid dimension (creates an N x N grid).
    minDroplets : int
        Minimum number of data points required in a grid cell for it to be shown.
    cmap : matplotlib.colors.Colormap
        Colormap to use for shading the grid.
    vlims : Optional[Tuple[float, float]]
        Value limits for colormap scaling. If None, min/max of data is used.
    ax : matplotlib.axes.Axes
        Matplotlib axis object for plotting.
    
    Returns:
    -------
    matplotlib.axes.Axes
        The updated axis object with the shaded grid plot.
    """
    # Creating the grid
    x_edges = np.linspace(x_lim[0], x_lim[1], N+1)
    y_edges = np.linspace(y_lim[0], y_lim[1], N+1)
    
    # Convert Boolean column to int if necessary
    if df[avg_col].dtype == bool:
        df = df.copy()  # Create a copy to avoid modifying the original dataframe
        df['temp_avg_col'] = df[avg_col].astype(int)
    else:
        df = df.copy()  # Create a copy to avoid modifying the original dataframe
        df['temp_avg_col'] = df[avg_col]
    
    # Initialize a matrix for the average values
    avg_values = np.zeros((N, N))
    
    # Calculate the average value for each square
    for i in range(N):
        for j in range(N):
            points_in_square = df[
                (df[x_col] >= x_edges[i]) & (df[x_col] < x_edges[i+1]) &
                (df[y_col] >= y_edges[j]) & (df[y_col] < y_edges[j+1])
            ]
            if len(points_in_square) > minDroplets:
                avg_values[i, j] = points_in_square['temp_avg_col'].mean()
            else:
                avg_values[i, j] = np.nan
    
    # Determine color map limits based on avg_col data type
    if df[avg_col].dtype == bool:
        vmin, vmax = 0, 1
    elif vlims is not None:
        vmin, vmax = vlims[0], vlims[1]
    else:
        vmin, vmax = np.nanmin(avg_values), np.nanmax(avg_values)
    
    # Plotting
    for i in range(N):
        for j in range(N):
            if not np.isnan(avg_values[i, j]):
                color = cmap((avg_values[i, j] - vmin) / (vmax - vmin))
                ax.fill_betweenx([y_edges[j], y_edges[j+1]], x_edges[i], x_edges[i+1], color=color)
    
    # Set up the plot
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    # Create a colorbar
    fig = ax.figure
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f'Average {avg_col}')
    
    return ax

def plot_dominance_comparison(
    response_list: List[List[float]],
    modulator_values_list: List[List[float]],
    experiment_labels: List[str] = None,
    slice_labels: List[str] = None,
    slice_values: List[float] = None,
    slice_window_size: float = None,
    modulator_label: str = "Modulator",
    response_component_label: str = "Component",
    figsize: Tuple[int, int] = (6, 3),
    custom_colors: Optional[List] = None,
    confidence_level: float = 0.95,
    show_errors: bool = True,
    response_errors_list: List[List[float]] = None,  # Moved to match function call
    legend_loc: str = 'best'
) -> plt.Figure:
    """
    Plot a comparison of dominance responses across different experiments or slices with confidence intervals.
    
    Parameters:
    ----------
    response_list : List[List[float]]
        List of response values (typically slopes) for each experiment/slice.
    modulator_values_list : List[List[float]]
        List of modulator values for each experiment/slice.
    experiment_labels : List[str], optional
        Labels for each experiment. Used if slice_labels is None.
    slice_labels : List[str], optional
        Labels for slices if using sliced data. Defaults to None.
    slice_values : List[float], optional
        Numeric values for each slice. Required if slice_labels is provided.
    slice_window_size : float, optional
        Width of each slice window. Required if slice_labels is provided.
    modulator_label : str, optional
        Label for the modulator axis. Defaults to "Modulator".
    response_component_label : str, optional
        Label for the response component. Defaults to "Component".
    figsize : Tuple[int, int], optional
        Figure size. Defaults to (10, 6).
    custom_colors : Optional[List], optional
        Custom color palette. If None, a default colormap is used.
    confidence_level : float, optional
        Confidence level for error bands (e.g., 0.95 for 95%). Defaults to 0.95.
    show_errors : bool, optional
        Whether to show error bands. Defaults to True.
    response_errors_list : List[List[float]], optional
        List of standard errors for responses. If None, no error bands are shown.
    legend_loc : str, optional
        Location of the legend. Defaults to 'best'.
    
    Returns:
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    if custom_colors is None:
        colors = sns.color_palette('viridis', len(response_list))
    else:
        colors = custom_colors
    
    # Choose appropriate labels
    if experiment_labels is None:
        if slice_labels is not None and slice_values is not None and slice_window_size is not None:
            # Use slice-based labels
            experiment_labels = [f'{slice_labels[0]} = {val:.2f}-{val+slice_window_size:.2f}' 
                              for val in slice_values]
        else:
            # Use generic labels
            experiment_labels = [f"Experiment {i+1}" for i in range(len(response_list))]
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate appropriate z-score for the confidence level
    # For 95% confidence, z ≈ 1.96
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    for i, (responses, modulator_values, label) in enumerate(zip(response_list, modulator_values_list, experiment_labels)):
        # Convert responses to dominance values (1 - response)
        dominance_values = [1 - r for r in responses]
        
        # Plot dominance curve
        line = ax.plot(
            modulator_values, 
            dominance_values,
            '-o', 
            color=colors[i],
            label=label,
            linewidth=2,
            markersize=5
        )
        
        # Add confidence intervals if error values provided and show_errors is True
        if response_errors_list is not None and show_errors and i < len(response_errors_list):
            errors = response_errors_list[i]
            if len(errors) == len(responses):
                # Calculate upper and lower confidence bounds
                lower_bound = [max(0, 1 - r - z_score * err) for r, err in zip(responses, errors)]
                upper_bound = [min(1, 1 - r + z_score * err) for r, err in zip(responses, errors)]
                
                # Add shaded confidence bands
                ax.fill_between(
                    modulator_values,
                    lower_bound,
                    upper_bound,
                    color=colors[i],
                    alpha=0.2
                )
    
    # Set up the plot
    ax.set_xlabel(f'{modulator_label} Concentration')
    ax.set_ylabel(f'{response_component_label} Dominance')
    # ax.set_ylim(-0.1, 1.1)  # Dominance typically ranges from 0 to 1
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc)
    
    # Add confidence level to title if showing errors
    title = "Comparative Dominance Analysis"
    if show_errors and response_errors_list is not None:
        title += f" with {int(confidence_level*100)}% Confidence Intervals"
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def dominance_sweep(
    df: pd.DataFrame,
    protein_conc_col: str,
    protein_min: float,
    protein_max: float,
    modulator_conc_col: str,
    modulator_min: float,
    modulator_max: float,
    condensate_label: str,
    numberWindows: int,
    numberResponsePlots: int,
    minimumDataThreshold: int,
    modulator_label: str = 'Modulator',
    protein_label: str = 'Protein',
    phase_sep_col: str = 'Phase separated',
    dilute_intensity_col: str = 'dilute_intensity',  # Raw dilute intensity
    total_intensity_col: str = 'total_intensity',    # Raw total intensity
    plot_results: bool = True,
    tieline_color_demixed: List = None,
    tieline_color_mixed: List = None
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate dominance of a component across a range of modulator concentrations.
    
    This function divides the modulator concentration range into windows, and for each window,
    fits linear regressions to phase-separated and non-phase-separated data points.
    The slopes of these lines represent the relationship between total intensity
    and dilute intensity, which is used to calculate dominance.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the phase separation data.
    protein_conc_col : str
        Column name for the protein concentration.
    protein_min : float
        Minimum value for protein concentration in plots.
    protein_max : float
        Maximum value for protein concentration in plots.
    modulator_conc_col : str
        Column name for the modulator concentration.
    modulator_min : float
        Minimum value for modulator concentration to consider.
    modulator_max : float
        Maximum value for modulator concentration to consider.
    condensate_label : str
        Label identifying the condensate type in the dataframe.
    numberWindows : int
        Number of windows to divide the modulator range into.
    numberResponsePlots : int
        Number of representative plots to show for visualization.
    minimumDataThreshold : int
        Minimum number of data points required for fitting.
    modulator_label : str, optional
        Label for the modulator in plots. Defaults to 'Modulator'.
    protein_label : str, optional
        Label for the protein in plots. Defaults to 'Protein'.
    phase_sep_col : str, optional
        Column name indicating phase separation state. Defaults to 'Phase separated'.
    dilute_intensity_col : str, optional
        Column name for raw dilute intensity. Defaults to 'dilute_intensity'.
    total_intensity_col : str, optional
        Column name for raw total intensity. Defaults to 'total_intensity'.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.
    tieline_color_demixed : List, optional
        Colors for demixed (phase-separated) points. Defaults to None.
    tieline_color_mixed : List, optional
        Colors for mixed (non-phase-separated) points. Defaults to None.
    
    Returns:
    -------
    Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]
        Tuple containing:
        - PhaseSepResponses: List of slopes for phase-separated points
        - NonPhaseSepResponses: List of slopes for non-phase-separated points
        - PhaseSepModulatorValues: List of modulator values for phase-separated points
        - NonPhaseSepModulatorValues: List of modulator values for non-phase-separated points
        - PhaseSepResponseErrors: List of standard errors for phase-separated responses
        - NonPhaseSepResponseErrors: List of standard errors for non-phase-separated responses
    """
    # Set default color schemes if not provided
    if tieline_color_demixed is None:
        phase_sep_color_palette = sns.color_palette('RdBu_r', 5)
        tieline_color_demixed = [phase_sep_color_palette[-1], phase_sep_color_palette[-3]]
    
    if tieline_color_mixed is None:
        phase_sep_color_palette = sns.color_palette('RdBu_r', 5)
        tieline_color_mixed = [phase_sep_color_palette[2], phase_sep_color_palette[0]]
    
    # Create figure if plotting is enabled
    if plot_results:
        fig, ax = plt.subplots(1, numberResponsePlots+1, figsize=(3*(numberResponsePlots+1), 3))
        # Make sure ax is always an array, even with only one plot
        if numberResponsePlots == 0:
            ax = np.array([ax])
        elif not isinstance(ax, np.ndarray):
            ax = np.array([ax])
    
    modulatorWindows = np.linspace(modulator_min, modulator_max, numberWindows)
    modulatorWindowSize = modulatorWindows[1] - modulatorWindows[0]

    plotCount = 0
    # Fix calculation of how many windows to skip between plots
    if numberResponsePlots > 0:
        ResponsePlotEveryN = numberWindows // numberResponsePlots
        if ResponsePlotEveryN == 0:  # Ensure at least 1 window between plots
            ResponsePlotEveryN = 1
    else:
        ResponsePlotEveryN = numberWindows + 1  # Ensures no response plots if numberResponsePlots = 0
    
    # Lists to store results
    PhaseSepResponses: List[float] = []
    PhaseSepResponseErrors: List[float] = []
    NonPhaseSepResponses: List[float] = []
    NonPhaseSepResponseErrors: List[float] = []
    PhaseSepModulatorValues: List[float] = []
    NonPhaseSepModulatorValues: List[float] = []

    for i, modulatorWindow in enumerate(modulatorWindows):
        # Filter data for the current modulator window
        mask = (
            (df[modulator_conc_col] > modulatorWindow) &
            (df[modulator_conc_col] < modulatorWindow + modulatorWindowSize) &
            (df['Condensate label'] == condensate_label)
        )

        temp_df = df[mask]

        # Separate phase-separated and non-phase-separated points
        slicedf_ps = temp_df[temp_df[phase_sep_col]]
        slicedf_nops = temp_df[~temp_df[phase_sep_col]]

        # Analyze phase-separated points
        if len(slicedf_ps) > minimumDataThreshold:     
            try:
                # Changed this to use raw intensity values
                slope_ps, intercept_ps, r_ps, p_ps, se_ps = stats.linregress(
                    slicedf_ps[total_intensity_col],
                    slicedf_ps[dilute_intensity_col]
                )
                
                PhaseSepModulatorValues.append(modulatorWindow + modulatorWindowSize/2) 
                PhaseSepResponses.append(slope_ps)
                PhaseSepResponseErrors.append(se_ps)
                
            except ValueError:
                pass

        # Analyze non-phase-separated points
        if len(slicedf_nops) > minimumDataThreshold:
            try:
                # Changed this to use raw intensity values  
                slope_nops, intercept_nops, r_nops, p_nops, se_nops = stats.linregress(
                    slicedf_nops[total_intensity_col],
                    slicedf_nops[dilute_intensity_col]
                )
    
                NonPhaseSepModulatorValues.append(modulatorWindow + modulatorWindowSize/2) 
                NonPhaseSepResponses.append(slope_nops)
                NonPhaseSepResponseErrors.append(se_nops)
            except ValueError:
                pass

        # Create visualizations at specified intervals - fixed plotting logic
        if plot_results and numberResponsePlots > 0 and i % ResponsePlotEveryN == 0 and plotCount < numberResponsePlots:
            # Check if there's enough data to plot
            if len(slicedf_ps) > 0 or len(slicedf_nops) > 0:
                current_ax = ax[plotCount+1] if len(ax) > plotCount+1 else ax[-1]
                
                # Plot phase-separated points if they exist
                if len(slicedf_ps) > 0:
                    current_ax.scatter(
                        x=slicedf_ps[total_intensity_col],
                        y=slicedf_ps[dilute_intensity_col],
                        c=tieline_color_demixed[0],
                        s=1,
                        alpha=0.5
                    )
                
                # Plot non-phase-separated points if they exist
                if len(slicedf_nops) > 0:
                    current_ax.scatter(
                        x=slicedf_nops[total_intensity_col],
                        y=slicedf_nops[dilute_intensity_col],
                        c=tieline_color_mixed[1],
                        s=1,
                        alpha=0.5
                    )
                
                current_ax.set_xlabel(f'Total {protein_label} Intensity')
                current_ax.set_ylabel(f'Dilute {protein_label} Intensity')
                current_ax.set_title(f'{modulator_label} = {(modulatorWindow + modulatorWindowSize/2):.1f} ± {modulatorWindowSize:.1f}')
                
                # Determine plot limits based on data range
                y_max = max(
                    slicedf_ps[dilute_intensity_col].max() if len(slicedf_ps) > 0 else 0, 
                    slicedf_nops[dilute_intensity_col].max() if len(slicedf_nops) > 0 else 0,
                    0.001  # prevent zeros
                )
                x_max = max(
                    slicedf_ps[total_intensity_col].max() if len(slicedf_ps) > 0 else 0, 
                    slicedf_nops[total_intensity_col].max() if len(slicedf_nops) > 0 else 0,
                    0.001  # prevent zeros
                )
                plot_max = max(y_max, x_max) * 1.1
                
                current_ax.set_ylim(0, plot_max)
                current_ax.set_xlim(0, plot_max)
                current_ax.set_aspect('equal')
                current_ax.plot([0, plot_max], [0, plot_max], linestyle='--', c='k')
                plotCount += 1

    # Plot the dominance summary
    if plot_results and len(PhaseSepResponses) > 0:
        # Check if there's at least one axis
        summary_ax = ax[0] if len(ax) > 0 else ax
        
        summary_ax.scatter(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponses],
            c='r',
            label='Demixed',
            s=10,
            alpha=1
        )
        
        if len(NonPhaseSepResponses) > 0:
            summary_ax.scatter(
                x=NonPhaseSepModulatorValues,
                y=[1 - x for x in NonPhaseSepResponses],
                c='k',
                label='Mixed',
                s=10,
                alpha=1
            )
        
        summary_ax.errorbar(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponses],
            c='r',
            linestyle='-',
            yerr=PhaseSepResponseErrors
        )
        
        if len(NonPhaseSepResponses) > 0 and len(NonPhaseSepResponseErrors) > 0:
            summary_ax.errorbar(
                x=NonPhaseSepModulatorValues,
                y=[1 - x for x in NonPhaseSepResponses],
                c='k',
                linestyle='-',
                yerr=NonPhaseSepResponseErrors
            )
                
        summary_ax.set_xlabel(f'{modulator_label}')
        summary_ax.set_ylabel(f'{protein_label} Dominance')
        summary_ax.set_ylim(-0.2, 1)
        summary_ax.set_xlim(modulator_min, modulator_max)
        summary_ax.legend()

    # Return standard errors as well
    return PhaseSepResponses, NonPhaseSepResponses, PhaseSepModulatorValues, NonPhaseSepModulatorValues, PhaseSepResponseErrors, NonPhaseSepResponseErrors

def piecewise_linear_redefined(
    x: np.ndarray,
    k1: float,
    delta_k: float,
    y0: float,
    x0: float
) -> np.ndarray:
    """
    Redefined piecewise linear function with two segments and the constraint that k1 < k2.
    
    Parameters:
    ----------
    x : np.ndarray
        Input x values.
    k1 : float
        Slope of the first line segment.
    delta_k : float
        Difference between k2 and k1, ensuring k1 < k2.
    y0 : float
        y-intercept of the first line segment.
    x0 : float
        x-value where the two line segments meet.
    
    Returns:
    -------
    np.ndarray
        Output y values.
    """
    k2 = k1 + delta_k  # Ensure k1 < k2
    return np.piecewise(
        x, 
        [x < x0, x >= x0], 
        [lambda x: k1 * x + y0, lambda x: k2 * (x - x0) + (k1 * x0 + y0)]
    )


def dominance_sweep_piecewiselinear(
    df: pd.DataFrame,
    protein_conc_col: str,
    protein_min: float,
    protein_max: float,
    modulator_conc_col: str,
    modulator_min: float,
    modulator_max: float,
    condensate_label: str,
    numberWindows: int,
    numberResponsePlots: int,
    minimumDataThreshold: int,
    initial_guesses: List[float],
    bounds: Tuple[List[float], List[float]],
    modulator_label: str = 'Modulator',
    protein_label: str = 'Protein',
    phase_sep_col: str = 'Phase separated',
    dilute_intensity_col: str = 'Normalised dilute intensity',
    plot_results: bool = True,
    tieline_color_demixed: List = None,
    tieline_color_mixed: List = None
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate dominance with piecewise linear fitting for more complex phase behavior.
    
    This is similar to dominance_sweep but uses a piecewise linear function with two
    different slopes to better model systems that may exhibit complex binding behavior.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the phase separation data.
    protein_conc_col : str
        Column name for the protein concentration.
    protein_min : float
        Minimum value for protein concentration in plots.
    protein_max : float
        Maximum value for protein concentration in plots.
    modulator_conc_col : str
        Column name for the modulator concentration.
    modulator_min : float
        Minimum value for modulator concentration to consider.
    modulator_max : float
        Maximum value for modulator concentration to consider.
    condensate_label : str
        Label identifying the condensate type in the dataframe.
    numberWindows : int
        Number of windows to divide the modulator range into.
    numberResponsePlots : int
        Number of representative plots to show for visualization.
    minimumDataThreshold : int
        Minimum number of data points required for fitting.
    initial_guesses : List[float]
        Initial parameter guesses for the piecewise linear fit [k1, delta_k, y0, x0].
    bounds : Tuple[List[float], List[float]]
        Lower and upper bounds for the fit parameters.
    modulator_label : str, optional
        Label for the modulator in plots. Defaults to 'Modulator'.
    protein_label : str, optional
        Label for the protein in plots. Defaults to 'Protein'.
    phase_sep_col : str, optional
        Column name indicating phase separation state. Defaults to 'Phase separated'.
    dilute_intensity_col : str, optional
        Column name for normalized dilute intensity. Defaults to 'Normalised dilute intensity'.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.
    tieline_color_demixed : List, optional
        Colors for demixed (phase-separated) points. Defaults to None.
    tieline_color_mixed : List, optional
        Colors for mixed (non-phase-separated) points. Defaults to None.
    
    Returns:
    -------
    Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]
        Tuple containing:
        - PhaseSepResponsesFit1: Slopes from first segment of piecewise fit
        - PhaseSepResponsesFit2: Slopes from second segment of piecewise fit
        - NonPhaseSepResponses: Slopes for non-phase-separated points
        - PhaseSepModulatorValues: Modulator values for phase-separated points
        - NonPhaseSepModulatorValues: Modulator values for non-phase-separated points
        - interceptxValues: x-values where the two line segments meet
    """
    # Set default color schemes if not provided
    if tieline_color_demixed is None:
        phase_sep_color_palette = sns.color_palette('RdBu_r', 5)
        tieline_color_demixed = [phase_sep_color_palette[-1], phase_sep_color_palette[-3]]
    
    if tieline_color_mixed is None:
        phase_sep_color_palette = sns.color_palette('RdBu_r', 5)
        tieline_color_mixed = [phase_sep_color_palette[2], phase_sep_color_palette[0]]
    
    # Create figure if plotting is enabled
    if plot_results:
        fig, ax = plt.subplots(1, numberResponsePlots+1, figsize=(3*(numberResponsePlots+1), 3))
    
    modulatorWindows = np.linspace(modulator_min, modulator_max, numberWindows)
    modulatorWindowSize = modulatorWindows[1] - modulatorWindows[0]

    plotCount = 0
    ResponsePlotEveryN = numberWindows / numberResponsePlots
    
    # Lists to store results
    PhaseSepResponsesFit1: List[float] = [] 
    PhaseSepResponsesFit2: List[float] = [] 
    PhaseSepResponsesFit1Errs: List[float] = [] 
    PhaseSepResponsesFit2Errs: List[float] = [] 
    NonPhaseSepResponses: List[float] = [] 
    PhaseSepModulatorValues: List[float] = []
    NonPhaseSepModulatorValues: List[float] = []
    interceptxValues: List[float] = []

    for i, modulatorWindow in enumerate(modulatorWindows):
        # Filter data for the current modulator window
        mask = (
            (df[modulator_conc_col] > modulatorWindow) &
            (df[modulator_conc_col] < modulatorWindow + modulatorWindowSize) &
            (df['Condensate label'] == condensate_label)
        )

        temp_df = df[mask]

        # Separate phase-separated and non-phase-separated points
        slicedf_ps = temp_df[temp_df[phase_sep_col]]
        slicedf_nops = temp_df[~temp_df[phase_sep_col]]

        # Analyze phase-separated points with piecewise linear fit
        if slicedf_ps.shape[0] > minimumDataThreshold:     
            try:
                params, params_covariance = curve_fit(
                    piecewise_linear_redefined,
                    np.asarray(slicedf_ps[protein_conc_col]),
                    np.asarray(slicedf_ps[protein_conc_col] * slicedf_ps[dilute_intensity_col]), 
                    p0=initial_guesses,
                    bounds=bounds
                )
                
                interceptxValues.append(params[-1])
                PhaseSepModulatorValues.append(modulatorWindow + modulatorWindowSize/2) 
                PhaseSepResponsesFit1.append(params[0])
                PhaseSepResponsesFit2.append(params[0] + params[1])
                
                standard_errors = np.sqrt(np.diag(params_covariance))
                PhaseSepResponsesFit1Errs.append(standard_errors[0])
                PhaseSepResponsesFit2Errs.append(np.sqrt(standard_errors[0]**2 + standard_errors[1]**2))
                
            except ValueError:
                pass

        # Analyze non-phase-separated points with simple linear fit
        if slicedf_nops.shape[0] > minimumDataThreshold:
            try:  
                slope_nops, intercept_nops, r_nops, p_nops, se_nops = stats.linregress(
                    slicedf_nops[protein_conc_col],
                    slicedf_nops[protein_conc_col] * slicedf_nops[dilute_intensity_col]
                )
    
                NonPhaseSepModulatorValues.append(modulatorWindow + modulatorWindowSize/2) 
                NonPhaseSepResponses.append(slope_nops)
            except ValueError:
                pass

        # Create visualizations at specified intervals
        if plot_results and i % ResponsePlotEveryN == 0 and plotCount < numberResponsePlots:
            ax[plotCount+1].scatter(
                x=slicedf_ps[protein_conc_col],
                y=slicedf_ps[protein_conc_col] * slicedf_ps[dilute_intensity_col],
                c=tieline_color_demixed[0],
                s=1,
                alpha=0.5
            )
            ax[plotCount+1].scatter(
                x=slicedf_nops[protein_conc_col],
                y=slicedf_nops[protein_conc_col] * slicedf_nops[dilute_intensity_col],
                c=tieline_color_mixed[1],
                s=1,
                alpha=0.5
            )
            
            ax[plotCount+1].set_xlabel(f'Total {protein_label}')
            ax[plotCount+1].set_ylabel(f'Dilute {protein_label}')
            ax[plotCount+1].set_title(f'{modulator_label} = {(modulatorWindow + modulatorWindowSize/2):.1f} ± {modulatorWindowSize:.1f}')
            ax[plotCount+1].set_ylim(protein_min, protein_max)
            ax[plotCount+1].set_xlim(protein_min, protein_max)
            ax[plotCount+1].set_aspect(1)
            ax[plotCount+1].plot([0, protein_max*1.2], [0, protein_max*1.2], linestyle='--', c='k')
            plotCount += 1

    # Plot the dominance summary
    if plot_results:
        ax[0].scatter(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponsesFit1],
            c='r',
            marker='o',
            label='Demixed Fit 1',
            s=3,
            alpha=1
        )
        ax[0].errorbar(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponsesFit1],
            c='r',
            linestyle=' ',
            yerr=PhaseSepResponsesFit1Errs
        )
        
        ax[0].scatter(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponsesFit2],
            c='grey',
            marker='x',
            label='Demixed Fit 2',
            s=3,
            alpha=1
        )
        ax[0].errorbar(
            x=PhaseSepModulatorValues,
            y=[1 - x for x in PhaseSepResponsesFit2],
            c='grey',
            linestyle=' ',
            yerr=PhaseSepResponsesFit2Errs
        )
        
        ax[0].scatter(
            x=NonPhaseSepModulatorValues,
            y=[1 - x for x in NonPhaseSepResponses],
            c='k',
            label='Mixed',
            s=3,
            alpha=1
        )

        ax[0].set_xlabel(f'{modulator_label}')
        ax[0].set_ylabel(f'{protein_label} Dominance')
        ax[0].set_ylim(-0.2, 1)
        ax[0].set_xlim(modulator_min, modulator_max)
        
        red_patch = mpatches.Patch(color='r', label='Low Conc.')
        grey_patch = mpatches.Patch(color='grey', label='High Conc.')
        black_patch = mpatches.Patch(color='k', label='Mixed')
        
        ax[0].legend(handles=[red_patch, grey_patch, black_patch])

    return (
        PhaseSepResponsesFit1,
        PhaseSepResponsesFit2,
        NonPhaseSepResponses,
        PhaseSepModulatorValues,
        NonPhaseSepModulatorValues,
        interceptxValues
    )


def find_common_frames_from_window(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_min: float,
    x_size: float,
    y_min: float,
    y_size: float
) -> pd.Series:
    """
    Find the most common frames within a specified window in the phase space.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    x_col : str
        Column name for x-coordinate.
    y_col : str
        Column name for y-coordinate.
    x_min : float
        Minimum x-value of the window.
    x_size : float
        Width of the window in x-dimension.
    y_min : float
        Minimum y-value of the window.
    y_size : float
        Height of the window in y-dimension.
    
    Returns:
    -------
    pd.Series
        Series containing the most common frame indices in the window.
    """
    mask = (
        (df[x_col] > x_min) & 
        (df[x_col] < x_min + x_size) & 
        (df[y_col] > y_min) & 
        (df[y_col] < y_min + y_size)
    )
    temp = df[mask]

    return temp['continuous_frame'].mode()

