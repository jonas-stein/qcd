import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_box_plot(data: list, data_labels=None, save_plot=False, type="", x_axis="", y_axis="Deviation"):
    """
    Draws (and saves) a boxplot based on the input data.

    Parameters
    ----------
    data : list
        All data to be plotted. Form: [data_1, ...] with data_i = [value_i1, ...].
        Usually contains the deviations of the self-found results from the best known solution.
    data_labels : list
        A list of strings containing labels for the specified data.
    save_plot : bool
        Specifies whether or not the resulting boxplot should be save to a file.
        Plots will be saved in the folder "Plots".
        Filename: boxplot-YYYY-MM-DD-HH-MM-SS-MILSEC.pdf
    type : String
        Only used for the name of the file should the plot get saved.
    x_axis : String
        The text for the x axis.
    y_axis : String
        The text for the y axis.
    """
    plt.style.use(os.path.join(os.getcwd(), "Plot Styles", "PaperSingleColumnFig.mplstyle"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if data_labels is None:
        data_labels = ['Dataset-' + str(i) for i in range(len(data))]

    # ax.axis([0.5,len(data_labels)+0.5,-0.05,1.05])
    max_value = 0
    for i in range(len(data)):
        temp_max = max(list(data[i]))
        if temp_max > max_value:
            max_value = temp_max

    print(max_value)
    # ax.axis([0.5,len(data_labels)+0.5,-0.05*max(1,max(data[max_index])),max(1,max(data[max_index]))+0.05*max(1,max(data[max_index]))])
    ax.axis([0.5, len(data_labels) + 0.5, -0.05 * max(1, max_value), max(1, max_value) + 0.05 * max(1, max_value)])

    line_width = 0.3
    marker_size = 4
    ax.boxplot(data, labels=data_labels, medianprops=dict(color='k', linewidth=line_width),
               boxprops=dict(linewidth=line_width),
               whiskerprops=dict(linestyle='--', dashes=(5, 13), linewidth=line_width),
               capprops=dict(linewidth=line_width),
               widths=[0.4] * len(data),
               flierprops=dict(marker='o', markerfacecolor='w', markersize=0,  # marker_size/2,
                               markeredgewidth=line_width, markeredgecolor='red')
               )

    for i in range(len(data)):
        dataset_points = data[i]
        ax.scatter(np.full_like(dataset_points, i + 1), dataset_points, marker='o', color='w', edgecolors='k',
                   linewidth=line_width, s=marker_size)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if save_plot is True:
        type = type.replace(" ", "_")
        filename = "boxplot-" + type + "-" + str(datetime.datetime.now())
        filename = filename.replace(" ", "--")
        filename = filename.replace(".", "-")
        filename = filename.replace(":", "-")
        filename += ".pdf"

        plots_path = os.path.join(os.getcwd(), "Plots")
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        filepath = os.path.join(plots_path, filename)

        plt.savefig(filepath, format="pdf", bbox_inches='tight', pad_inches=0)


def combine_box_plots(plots: list, save_plot=False):
    """
    Combines multiple .csv files for boxplots to one .pdf file with all plots.

    Parameters
    ----------
    plots : list of str
        A list of strings with the filepath for each .csv file
    save_plot : bool
        Determines whether the resulting plot should get saved.
    """
    plt.style.use(os.path.join(os.getcwd(), "Plot Styles", "PaperThreePlotsFig.mplstyle"))

    fig = plt.figure()
    for i in range(len(plots)):
        filepath = os.path.join(os.getcwd(), "Plots", plots[i])

        type = filepath.split("boxplot-", 1)[1]
        type = type.replace("--", ".")
        type = type.split("-", 1)[0]
        type = type.replace("_", " ")

        filepath = os.path.join(os.getcwd(), "Plots", filepath)
        filepath = filepath + ".csv"

        data = pd.read_csv(filepath)

        data_labels = data.columns.to_list()

        results = []
        for label in data_labels:
            results.append(data[label].tolist())

        ax = fig.add_subplot(130 + i + 1)
        ax2 = ax.twinx()

        if data_labels is None:
            data_labels = ['Dataset-' + str(i) for i in range(len(data))]

        ax.axis([0.5, len(data_labels) + 0.5, -0.05, 1.05])
        if results[2]:
            ax2.axis([0.5, len(data_labels) + 0.5, -0.05 * max(1, max(results[2])),
                      max(1, max(results[2])) + 0.05 * max(1, max(results[2]))])
        else:
            ax2.axis([0.5, len(data_labels) + 0.5, -0.05, 1.05])

        line_width = 0.075
        marker_size = 3
        ax.boxplot(results, labels=data_labels, medianprops=dict(color='k', linewidth=line_width),
                   boxprops=dict(linewidth=line_width),
                   whiskerprops=dict(linestyle='--', dashes=(5, 13), linewidth=line_width),
                   capprops=dict(linewidth=line_width),
                   widths=[0.4] * len(results),
                   flierprops=dict(marker='o', markerfacecolor='w', markersize=0,  # marker_size/2,
                                   markeredgewidth=line_width, markeredgecolor='red')
                   )

        for i in range(len(results)):  # TODO @Domi check i (already defined in outer for loop)
            dataset_points = results[i]
            ax.scatter(np.full_like(dataset_points, i + 1), dataset_points, marker='o', color='w', edgecolors='k',
                       linewidth=line_width, s=marker_size)
            if i == len(results) - 1:
                ax2.scatter(np.full_like(dataset_points, i + 1), dataset_points, marker='o', color='w', edgecolors='k',
                            linewidth=line_width, s=marker_size)

        ax.set_xlabel(type, fontsize=5.5)
        ax.set_ylabel('Accuracy', fontsize=4)
        ax2.set_ylabel('Number of bijective violations', fontsize=3.5)

    if save_plot is True:
        filename = "combined_boxplot-" + str(datetime.datetime.now())
        filename = filename.replace(" ", "--")
        filename = filename.replace(".", "-")
        filename = filename.replace(":", "-")
        filename += ".pdf"

        plots_path = os.path.join(os.getcwd(), "Plots")
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        filepath = os.path.join(plots_path, filename)

        plt.savefig(filepath, format="pdf", bbox_inches='tight', pad_inches=0)


def create_plot_from_file(filepath):
    """
    Creates a plot from the chosen .csv file. The plot will be saved as a pdf.
    Works only for .csv files in the folder ../Plots

    Parameters
    ----------
    filepath : String
        The path to the .csv file created through create_plot_file()

    Returns
    -------
    NOTHING
    """

    save_plot = True
    type = filepath.split("boxplot-", 1)[1]
    type = type.split("-", 1)[0]

    filepath = os.path.join(os.getcwd(), "Plots", filepath)

    data = pd.read_csv(filepath)

    header_labels = data.columns.to_list()

    deviation = []
    for header in header_labels:
        deviation.append(data[header].tolist())

    create_box_plot(deviation, header_labels, save_plot, type)
    print("Plot successfully created.")


def create_plot_csv_file(type, data: list, data_labels=None):
    """
    Creates a file in the current path ".../Plots/filename.txt"
    The filename consists of the current time and the method "type" that was used to create the plot.

    Parameters
    ----------
    type : String
        Becomes part of the filename to describe the plot
    data : list
        All data to be plotted. Form: [data_1, ...] with data_i = [value_i1, ...].
        Usually contains the deviations of the self-found results from the best known solution.
    data_labels : list
        A list of strings containing labels for the specified data.

    Returns
    -------
    NOTHING
    """

    filename = "boxplot-" + type + "-" + str(datetime.datetime.now())
    filename = filename.replace(" ", "--")
    filename = filename.replace(".", "--")
    filename = filename.replace(":", "-")
    # print("filename: " + filename)

    # create folder Plots
    plots_path = os.path.join(os.getcwd(), "Plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    filepath = os.path.join(plots_path, filename + ".csv")

    data_frame = pd.DataFrame(data)
    data_frame = data_frame.T
    data_frame.columns = data_labels
    data_frame.to_csv(filepath, index=False)

    return filename


def create_seed_file(type=""):
    """
    Creates a file in the current path ".../Seeds/filename.txt"
    The filename consists of the current time, the difficulty "diff" of the graphs and the method "type" of how the graph was analyzed.

    Parameters
    ----------
    type : String
        Becomes part of the filename to describe how the graph was analyzed

    Returns
    -------
    filepath : String
        The path of the created file
    """

    filename = "seeds-" + str(datetime.datetime.now())
    filename = filename.replace(" ", "--" + type + "--")
    filename = filename.replace(".", "--")
    filename = filename.replace(":", "-")
    # print("filename: " + filename)

    # create folder Seeds
    seeds_path = os.path.join(os.getcwd(), "Plots", "Seeds")
    if not os.path.exists(seeds_path):
        os.makedirs(seeds_path)

    filepath = os.path.join(seeds_path, filename + ".txt")
    with open(filepath, "w") as f:
        pass

    return filepath


def write_seed_file(filepath, diff, columns: list):
    """
    Writes the header line into the file "filepath" with the headers "diff" followed by each element of "columns".

    Parameters
    ----------
    filepath : str
        Becomes part of the filename to describe the complexity of the graph and headline of the first column
    diff : str
        Becomes part of the filename to describe how the graph was analyzed
    columns : list
        The labels for additional columns to save the results of each seed

    Returns
    -------
    NOTHING
    """
    with open(filepath, "a") as f:
        f.write(f"\n{diff:15}")
        for column in columns:
            f.write(f"{str(column):20}")
        f.write("\n")
