import os
import warnings
import math

import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from .utilities import colormap_from_str
from .plot_tree_patch import plot_tree_patched
from .utilities import rule_print_inline, custom_axes_limit
from .utilities import custom_formatter

IS_CI = os.environ.get("CI") == "true"


class InteractPoint:  # Object containg all information of a point
    def __init__(self, name, pos, color, size, shape, cluster_memb=None, value=None):
        self.name = str(name)  # string
        self.pos = pos  # list or tuple with two floats
        self.color = color  # list or tuple with 4 integers from 0 to 255 (rgba)
        # self.edges=edges                   #string (default: "black"), customisation not implemented
        self.size = size  # positive integer
        self.shape = shape  # string
        self.cluster_memb = cluster_memb  # string or None
        self.value = value  # float or None (learner predicted value)


class InteractPlot:  # Object containing all information of a plot
    def __init__(self, name, points, clustered=False, xlabel="PC1", ylabel="PC2"):
        self.name = str(name)  # string (should be unique)
        self.points = points  # list of InteractPoints objects
        self.clustered = clustered  # Boolean
        self.xlabel = xlabel  # string
        self.ylabel = ylabel  # string


def make_interactive_plot(plots, temp_files_dir, plot_size=700, other_inputs=None, max_depth=None):
    """
    Generates an interactive Graphical User Interface with (default) two plots, offering the
    the possibility to interact with them.

    Parameters:
    -----------
    plots : list
        A list of interactable_plot objects that contain the points and plotting information.
    plot_size : int, optional, default=700
        The size of the plot in pixels.
    other_inputs : object, optional
        An object containing additional inputs for the user, such as a fitted classifier and sample data.
    max_depth : int, optional
        The maximum depth for displaying the decision tree.

    Returns:
    --------
    list
        A list of interactable_plot objects for further handling and testing.

    """
    border_size = 0.1  # Proportion of borders around the plot.
    dpg.destroy_context()
    dpg.create_context()

    # DPG set theme
    lighttheme = create_theme_imgui_light()
    dpg.bind_theme(lighttheme)

    border_size = 0.1  # Proportion of borders around the plot.
    fullspacing = 8
    # dpg.destroy_context()
    # dpg.create_context()

    # Load and set up tree image
    # create image to load:
    # Activated when mouse is left clicked on a DPG object
    # only at this stage, the tree index is known,
    # as it corresponds to the point.hovered at the moment
    def mouse_click_left_callback():

        # Check all points to see if they are hovered
        for plot in plots:
            for point in plot.points:
                if point.hovered:
                    assert other_inputs is not None
                    # the other_inputs is supposed to be a TreeExtractor instance

                    my_clf = (
                        other_inputs.clf
                    )  # rf estimator here. Is this compatible with the Wrapper? Probably not
                    if hasattr(my_clf, "feature_names_in_"):
                        feature_names = my_clf.feature_names_in_
                    else:
                        feature_names = [f"X{i}" for i in range(my_clf.n_features_in_)]
                    sample_index = other_inputs.sample.index[0]
                    tree_index = int(point.name)
                    the_tree = my_clf[tree_index]

                    rule_print_inline(the_tree, other_inputs.sample)

                    tree_name_png = "Tree_" + str(tree_index) + ".png"

                    if max_depth is not None:
                        real_plot_leaves = max(the_tree.tree_.n_leaves, 2 ** (max_depth - 1))
                        real_plot_depth = min(the_tree.tree_.max_depth, max_depth)
                    else:
                        real_plot_leaves = the_tree.tree_.n_leaves
                        real_plot_depth = the_tree.tree_.max_depth

                    smart_width = 1 + 0.4 * real_plot_leaves
                    smart_width = int(smart_width)
                    smart_height = int(real_plot_depth + 1)
                    if (
                        my_clf.n_outputs_ > 3
                    ):  # create extra vertical space for multi-output trees, since output is printed over multiple lines
                        smart_height = int(smart_height * (0.92 + 0.08 * (my_clf.n_outputs_)))

                    plt.subplots(figsize=(smart_width, smart_height))

                    plot_tree_patched(
                        the_tree, max_depth=max_depth, feature_names=feature_names, fontsize=8
                    )

                    plt.title(
                        "Tree %i for sample index %i" % (tree_index, sample_index),
                        fontsize=10 + int(1.3 * real_plot_depth),
                    )
                    plt.savefig(os.path.join(temp_files_dir, tree_name_png))

                    if plt.isinteractive():
                        plt.show()  # Show selected tree in console, if interactive
                    # now that the needed tree image is created, we can load it
                    # generates provisional plot: load tree_name_png, or "trial" file:
                    # "PL"+plot.name+"_"+point.name+".png"
                    IMwidth, IMheight, _, IMdata = dpg.load_image(
                        os.path.join(temp_files_dir, tree_name_png)
                    )
                    os.remove(os.path.join(temp_files_dir, tree_name_png))

                    with dpg.texture_registry(show=False):
                        treeplot = dpg.add_static_texture(IMwidth, IMheight, IMdata)

                    popup_window = dpg.add_window(
                        width=min(IMwidth + 4 * fullspacing, windowwidth),
                        height=min(IMheight + 5 * fullspacing, windowheight),
                        modal=True,
                        horizontal_scrollbar=True,
                    )

                    dpg.add_image(treeplot, width=IMwidth, height=IMheight, parent=popup_window)

    # DPG mouse click setup
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Left, callback=mouse_click_left_callback
        )

    # Activated every frame ## IS THIS EVEN USED? CHECK!
    def point_draw_and_interact(sender, app_data, user_data):

        # Calculates positions in pixel space
        _helper_data = app_data[0]
        transformed_x = app_data[1]
        transformed_y = app_data[2]
        mouse_x_pixel_space = _helper_data["MouseX_PixelSpace"]
        mouse_y_pixel_space = _helper_data["MouseY_PixelSpace"]

        # Renaming for legibility
        plot = user_data
        points = plot.points

        # Deletes the previous frame
        # Slot = 2 specifies that only the dpg.draw objects must be deleted.
        dpg.delete_item(sender, children_only=True, slot=2)
        dpg.configure_item(sender, tooltip=False)

        # Goes through all points to draw them and to see if they are hovered
        for i, point in enumerate(points):

            # Draws point shape
            if point.shape == "circle":
                dpg.draw_circle(
                    [transformed_x[i], transformed_y[i]],
                    point.size,
                    fill=point.color,
                    parent=sender,
                )

            elif point.shape == "triangle":

                sin_size = 0.500 * point.size
                cos_size = 0.866 * point.size

                dpg.draw_triangle(
                    [transformed_x[i], transformed_y[i] - point.size],
                    [transformed_x[i] + cos_size, transformed_y[i] + sin_size],
                    [transformed_x[i] - cos_size, transformed_y[i] + sin_size],
                    fill=point.color,
                    parent=sender,
                )

            elif point.shape == "square":
                half_diagonal = 0.707 * point.size
                dpg.draw_rectangle(
                    [transformed_x[i] - half_diagonal, transformed_y[i] - half_diagonal],
                    [transformed_x[i] + half_diagonal, transformed_y[i] + half_diagonal],
                    fill=point.color,
                    parent=sender,
                )

            elif point.shape == "star":

                N_VERTECES = 5
                outer_radius = point.size * 1.1
                inner_radius = outer_radius * 0.4

                # Calculate the radial cooridnates for each vertex of the star
                outer_angles = [
                    math.radians(360 / N_VERTECES * i - 90 / N_VERTECES) for i in range(N_VERTECES)
                ]
                inner_angles = [
                    math.radians(360 / N_VERTECES * i + 90 / N_VERTECES) for i in range(N_VERTECES)
                ]

                # Calculate the coordinates of each vertex
                vertices = []
                for j in range(N_VERTECES):
                    outer_vertex = [
                        transformed_x[i] + outer_radius * math.cos(outer_angles[j]),
                        transformed_y[i] + outer_radius * math.sin(outer_angles[j]),
                    ]
                    inner_vertex = [
                        transformed_x[i] + inner_radius * math.cos(inner_angles[j]),
                        transformed_y[i] + inner_radius * math.sin(inner_angles[j]),
                    ]
                    vertices.extend([outer_vertex, inner_vertex])

                # Draw the star using the calculated vertices
                dpg.draw_polygon(vertices, fill=point.color, parent=sender)

            else:
                raise ValueError(f"Shape {point.shape} not supported")

            # Checks if cursor close enough for point to be considered hovered
            x_dif = transformed_x[i] - mouse_x_pixel_space
            y_dif = transformed_y[i] - mouse_y_pixel_space
            norm_squared = x_dif**2 + y_dif**2
            if norm_squared <= point.size**2:
                point.hovered = True
                dpg.configure_item(sender, tooltip=True)

                # Adds extra information to tooltip depending on point type
                tool_tip_text = "hovered Point: " + point.name
                if point.cluster_memb is not None:
                    tool_tip_text += "\n" + "Cluster: " + point.cluster_memb
                if point.value is not None:
                    tool_tip_text += "\n" + "Value: " + point.value

                # Shows tooltip when point is hovered
                dpg.set_value("TooltipText" + plot.name, tool_tip_text)
            else:
                point.hovered = False

    # Function that draws the average position of the different clusters
    def cluster_average(sender, app_data):
        size = 9
        x = app_data[1]
        y = app_data[2]

        # Reset previous frame
        dpg.delete_item(
            sender, children_only=True, slot=2
        )  # Deletes all drawn items (draw = slot 2)

        # Draws the average position
        for i in range(len(x)):
            dpg.draw_circle([x[i], y[i]], size, fill=[150, 150, 150, 255], parent=sender)
            dpg.draw_text([x[i] - size / 2, y[i] - size], clusters[i], size=size * 2, parent=sender)

    # A few useful extra variables
    plotamount = len(plots)
    addedwidth = 0
    # Now load images and calculate sizes and limits
    # Load in the temp_colourbar images, saving the desired size.

    for plot in plots:
        IMwidth, IMheight, _, IMdata = dpg.load_image(
            os.path.join(temp_files_dir, "temp_colourbar" + plot.name + ".png")
        )
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(IMwidth, IMheight, IMdata, tag="temp_colourbar" + plot.name)
        plot.colour_bar_width = round(plot_size * IMwidth / IMheight)
        addedwidth += plot.colour_bar_width

    windowwidth = plotamount * plot_size + (2 * plotamount + 1) * fullspacing + addedwidth
    windowheight = plot_size + 2 * fullspacing

    # Calculates x- and y-axis limits based on points
    for plot in plots:
        x_values = [point.pos[0] for point in plot.points]
        y_values = [point.pos[1] for point in plot.points]
        x_diff = max(x_values) - min(x_values)
        y_diff = max(y_values) - min(y_values)
        plot.x_axis_limits = [
            min(x_values) - x_diff * border_size,
            max(x_values) + x_diff * border_size,
        ]
        plot.y_axis_limits = [
            min(y_values) - y_diff * border_size,
            max(y_values) + y_diff * border_size,
        ]

    sample_index = other_inputs.sample.index[0]

    # DPG build windows
    with dpg.window(
        tag="MainWindow",
        width=windowwidth,
        label="Explaining sample "
        + str(sample_index)
        + ", close the window to continue with code execution",
        height=windowheight,
        no_title_bar=False,
        no_move=False,
        no_resize=True,
    ):
        with dpg.group(horizontal=True):
            for plot in plots:
                with dpg.plot(
                    tag="Tree representation plot" + plot.name, height=plot_size, width=plot_size
                ):
                    # create x-axis
                    dpg.add_plot_axis(dpg.mvXAxis, label=plot.xlabel, tag="x_axis" + plot.name)
                    dpg.set_axis_limits(
                        "x_axis" + plot.name, plot.x_axis_limits[0], plot.x_axis_limits[1]
                    )

                    # create y-axis
                    dpg.add_plot_axis(dpg.mvYAxis, label=plot.ylabel, tag="y_axis" + plot.name)
                    dpg.set_axis_limits(
                        "y_axis" + plot.name, plot.y_axis_limits[0], plot.y_axis_limits[1]
                    )

    viewport = dpg.create_viewport(title="Plot", width=1000, height=720)
    dpg.configure_viewport(
        viewport, height=windowheight + 5 * fullspacing, width=windowwidth + 2 * fullspacing
    )

    return plots


def plot_with_interface(
    plot_data_bunch,
    kmeans,
    input_method,  # a fitted bellatrex instance
    temp_files_dir,
    max_depth=None,
    colormap=None,
    clusterplots=(True, False),
):

    def shaper(in_shape):
        if in_shape is True:
            return "star"
        else:
            return "circle"

    def sizer(in_size):
        if in_size is True:
            return 16.0
        else:
            return 9.0

    # def edger(in_edge):
    #     return "black"
    # else:
    #     return (1,1,1,0.5) # semi-transparent white

    def rgbaconv(mpl_rgba):
        dpg_rgba = [i * 255 for i in mpl_rgba]
        return dpg_rgba

    # repeat PCA to 2 dimensions for projected trees
    # (original proj dimension can be > 2)
    # better: use original one and keep only the first two dims.
    PCA_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = PCA_fitted.transform(plot_data_bunch.proj_data)  # (lambda,2)

    # centers = PCA_fitted.transform(kmeans.cluster_centers_)
    cluster_memb = kmeans.labels_

    final_ts_idx = input_method.final_trees_idx

    is_final_candidate = [
        plot_data_bunch.index[i] in final_ts_idx for i in range(len(plot_data_bunch.index))
    ]
    custom_sizes = list(map(sizer, is_final_candidate))
    custom_shapes = list(map(shaper, is_final_candidate))
    # custom_edges = list(map(edger, is_final_candidate))

    plots = []

    color_map_left = cm.viridis  # default colormap for clustering plot
    color_map_left = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", [color_map_left(i) for i in range(color_map_left.N)], color_map_left.N
    )

    color_map_right = colormap_from_str(colormap)  # User custom colormap from object or from string

    # colormap = color_map_left if clustered == True else color_map_right

    for plotindex, clustered in enumerate(clusterplots):

        if clustered:
            fig, ax = plt.subplots(1, 1, figsize=(1, 4.5), dpi=120)

            # define the bins and normalize
            freqs = np.bincount(cluster_memb)
            if np.min(freqs) == 0:
                raise KeyError(
                    "There are empty clusters, the scatter and colorbar could differ in color shade"
                )
            norm_bins = list(np.cumsum(freqs))
            norm_bins.insert(0, 0)

            # transform list to array (ticks location needs arithmentic computation)
            norm_bins = np.array(norm_bins)

            # create label names
            labels = []
            for i in np.unique(cluster_memb):
                labels.append("cl.{:d}".format(i + 1))
            # normalizing color, prepare ticks, labels
            norm = mpl.colors.BoundaryNorm(norm_bins, color_map_left.N)
            tickz = norm_bins[:-1] + (norm_bins[1:] - norm_bins[:-1]) / 2

            cb1 = mpl.colorbar.Colorbar(
                ax,
                cmap=color_map_left,
                norm=norm,
                spacing="proportional",
                ticks=tickz,
                boundaries=norm_bins,
                format="%1i",
            )
            # label="cluster membership")
            cb1.ax.set_yticklabels(labels)  # vertically oriented colorbar

            ax.yaxis.set_ticks_position("left")
            norms = [norm(norm_bins[cluster_memb[i]]) for i in range(len(cluster_memb))]

        else:  # not the clustering plot, but the prediction/loss distribution:
            fig, ax = plt.subplots(1, 1, figsize=(1.2, 4.5), dpi=110)

            # binary, regression, survival case (single-output cases)
            if isinstance(plot_data_bunch.rf_pred, float) or plot_data_bunch.rf_pred.size == 1:

                is_binary = plot_data_bunch.set_up == "bin"

                plot_data_bunch.rf_pred = np.array(plot_data_bunch.rf_pred).squeeze()

                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.pred).min(),
                    np.array(plot_data_bunch.pred).max(),
                    plot_data_bunch.rf_pred,
                    is_binary,
                )

                norm_preds = mpl.colors.BoundaryNorm(
                    np.linspace(v_min, v_max, 256), color_map_right.N
                )

                ## add to colorbar a line corresponding to Bellatrex prediction
                # TODO: improve customFormatter function (ability to choose nb digits) and use that instead of np.round
                pred_tick = np.round(plot_data_bunch.rf_pred, 3)

                cb2 = mpl.colorbar.Colorbar(
                    ax, cmap=color_map_right, norm=norm_preds, label="RF pred: " + str(pred_tick)
                )

                plot_data_bunch.pred = np.array(
                    plot_data_bunch.pred
                ).squeeze()  # force shape: (n_trees,)

                # add tick in correspondence to the single trees prediction
                cb2.ax.plot([0, 1], [plot_data_bunch.pred] * 2, color="grey", linewidth=1)

                # add indicator (visually: >--<) in correspondence to the ensemble prediction
                cb2.ax.plot([0.02, 0.98], [pred_tick] * 2, color="black", linewidth=2.5, marker="P")

            # multi-output case, L2 losses instead of predictions
            else:
                # Blue for small losses, red for big losses
                # color_map_right = colormap_from_str('RdYlBu_r')
                color_map_right = colormap_from_str(
                    colormap
                )  # User custom colormap from object or from string

                v_min, v_max = custom_axes_limit(
                    np.array(plot_data_bunch.loss).min(),
                    np.array(plot_data_bunch.loss).max(),
                    force_in=None,
                    is_binary=False,
                )

                norm_preds = mpl.colors.BoundaryNorm(
                    np.linspace(v_min, v_max, 256), color_map_right.N
                )

                cb2 = mpl.colorbar.Colorbar(
                    ax,
                    cmap=color_map_right,
                    norm=norm_preds,
                    label=str(input_method.fidelity_measure) + " loss",
                )
                cb2.ax.plot([0, 1], [plot_data_bunch.loss] * 2, color="grey", linewidth=1)

            # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2g'))
            ticks_to_plot = ax.get_yticks()

            if np.abs(np.min(ticks_to_plot)) < 1e-3 and np.abs(np.max(ticks_to_plot)) > 1e-2:
                min_index = np.argmin(ticks_to_plot)
                ticks_to_plot[min_index] = 0
                ax.set_yticks(ticks_to_plot)

            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.minorticks_off()

            # single output case: float, or (1,)-shaped ndarray
            if isinstance(plot_data_bunch.rf_pred, float) or plot_data_bunch.rf_pred.size == 1:
                norms = [
                    norm_preds(float(plot_data_bunch.pred[i])) for i in range(len(cluster_memb))
                ]
            else:  # multi output case
                norms = [norm_preds(plot_data_bunch.loss[i]) for i in range(len(cluster_memb))]

        fig.tight_layout()
        fig.savefig(os.path.join(temp_files_dir, f"temp_colourbar{plotindex}.png"))
        plt.close(fig)  # prevent them from being shown in the console.

        # use appropriate colormap (right plot vs left plot)
        cmap_gui = color_map_left if clustered else color_map_right

        colours = [rgbaconv(cmap_gui(norms[i])) for i in range(len(plot_data_bunch.index))]

        points = []
        for j, index in enumerate(plot_data_bunch.index):
            points.append(
                InteractPoint(
                    index,
                    plottable_data[j],  # name and position
                    colours[j],
                    custom_sizes[j],  # size
                    # custom_edges[j],  # edge
                    custom_shapes[j],
                )  # shape
            )

            # adds extra information to the point for the tooltip
            if clustered:
                points[j].cluster_memb = str(
                    cluster_memb[j] + 1
                )  # +1 to change from 0-indexed to 1-indexed
            else:
                if isinstance(plot_data_bunch.pred[j], float):
                    points[j].value = (
                        f"{plot_data_bunch.pred[j]:.3f}"  # Show 3 numbers after decimal point
                    )
                elif isinstance(plot_data_bunch.loss[j], float):
                    points[j].value = (
                        f"{plot_data_bunch.loss[j]:.3f}"  # Show 3 numbers after decimal point
                    )
                else:
                    raise ValueError("expecting float, got {type(plot_data_bunch.loss)} instead")

        plots.append(InteractPlot(plotindex, points, clustered=clustered))

    # end of the enumerate(clusterplots) `for' loop (length 2 -> two plots)
    viewport_dict = make_interactive_plot(
        plots, temp_files_dir, plot_size=700, other_inputs=input_method, max_depth=max_depth
    )

    try:
        os.remove(os.path.join(temp_files_dir, "temp_colourbar0.png"))
        os.remove(os.path.join(temp_files_dir, "temp_colourbar1.png"))
    except (FileNotFoundError, UserWarning) as e:
        warnings.warn(
            f"{e}\nCould not delete temporary files correctly. Double check the "
            f"folder:\n {temp_files_dir}"
        )

    return plots  # Return the plots object for further handling
