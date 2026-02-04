import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

def plot_data(args):
    """Plot data from the provided files."""
    plt.style.use('ggplot')

    # Use the specified figure width and height from the arguments
    fig_width = args.fig_width
    fig_height = args.fig_height
    plt.rcParams["figure.figsize"] = [fig_width, fig_height]
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots()

    # Iterate over multiple files and plot each one with corresponding parameters
    for i, result_file in enumerate(args.data_files):
        data = np.genfromtxt(result_file, skip_footer=0, skip_header=0)
        color = args.colors[i] if args.colors else 'blue'
        linewidth = float(args.linewidths[i]) if args.linewidths else 1
        marker = args.markers[i] if args.markers else 'o'
        linestyle = args.linestyles[i] if args.linestyles else '-'
        label = args.labels[i] if args.labels else f"Curve {i+1}"

        ax.plot(data[:,int(args.x_data)], data[:,int(args.y_data)],
                linestyle=linestyle,
                marker=marker,
                linewidth=linewidth,
                color=color,
                label=label)

    # axis label
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.ticklabel_format(axis="both", style="plain")

    # axis scale
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)

    # legend
    ax.legend(fontsize=18)

    # title
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width * 0.975, box.height * 0.975])
    ax.set_title(args.title, y=1.04, fontsize=19)

    # save figure
    print(f"Figure has been saved to {args.plot_file}")
    fig.savefig(args.plot_file, dpi=300)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generic script for regular plots.")

    parser.add_argument("--data-files", nargs='+', type=str, help="Result files containing the data to plot.", required=True)
    parser.add_argument("--plot-file", type=str, help="Name of the file in which the figure will be saved.", default="plot.pdf")
    parser.add_argument("--title", type=str, help="Title of plot figure.", default="[TITLE]")
    parser.add_argument("--xlabel", type=str, help="Label for the x-axis.", default="[X LABEL]")
    parser.add_argument("--ylabel", type=str, help="Label for the y-axis.", default="[Y LABEL]")

    parser.add_argument("--x-data", type=str, help="Column index for the x-data.", default=0)
    parser.add_argument("--y-data", type=str, help="Column index for the y-data.", default=1)

    # Adding options to plot curves
    parser.add_argument("--colors", nargs='+', type=str, help="Colors for each curve.")
    parser.add_argument("--linewidths", nargs='+', type=str, help="Line widths for each curve.")
    parser.add_argument("--markers", nargs='+', type=str, help="Markers for each curve.")
    parser.add_argument("--linestyles", nargs='+', type=str, help="Line styles for each curve.")
    parser.add_argument("--labels", nargs='+', type=str, help="Labels for each curve.")

    # Adding options for axis scales
    parser.add_argument("--xscale", type=str, choices=["linear", "log", "symlog", "logit"], default="linear", help="Scale for the x-axis.")
    parser.add_argument("--yscale", type=str, choices=["linear", "log", "symlog", "logit"], default="linear", help="Scale for the y-axis.")

    # Adding options for figure width and height
    parser.add_argument("--fig-width", type=float, help="Width of the figure in inches.", default=10)
    parser.add_argument("--fig-height", type=float, help="Height of the figure in inches.", default=8)

    args = parser.parse_args()
    return args

def main():
    """Main function."""
    args = parse_arguments()
    plot_data(args)

if __name__ == "__main__":
    main()
