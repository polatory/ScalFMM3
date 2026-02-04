import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.option("-f", "--file", required=True, help="input csv file")
@click.option("-n", "--name", required=True, help="benchmark name")
def main(
    file: str,
    name: str,
):
    """Generate figure from a csv file"""

    df = pd.read_csv(file)

    if name == "accuracy":
        for ndim, group_ndim in df.groupby('ndim'):
            plt.figure()
            for (kernel_type, interp_type, tree_height), group in group_ndim.groupby(['kernel_type', 'interp_type', 'tree_height']):
                x = group['interp_order']
                y = group['error']
                plt.plot(x, y, label=f"k={kernel_type}-i={interp_type}-h={tree_height}")
                plt.yscale('log')
            plt.xlabel('Order')
            plt.ylabel('Relative norm-2 error')
            plt.title(f'Error vs. Order (d={ndim})')
            plt.legend()
            plt.savefig(f'scalfmm_accuracy_d={ndim}.pdf', bbox_inches='tight')

    elif name == "timeseq":
        for ndim, group_ndim in df.groupby('ndim'):
            plt.figure()
            for interp_type, group_interp in group_ndim.groupby('interp_type'):
                x = group_interp['size']
                y = group_interp['timefull_avg']
                plt.plot(x, y, label=f"i={interp_type}", marker='x')
                if interp_type == 1:
                    for xi, yi, height in zip(x, y, group_interp['tree_height']):
                        plt.text(xi, yi, str(height), fontsize=8, ha='right')
            kernel_type = group_interp['kernel_type'].iloc[0]
            interp_order = group_ndim['interp_order'].iloc[0]
            plt.xlabel('Size')
            plt.ylabel('Time (s)')
            plt.title(f'Time vs. Size (d={ndim},k={kernel_type},o={interp_order})')
            plt.legend()
            plt.savefig(f'scalfmm_timeseq_d={ndim}.pdf', bbox_inches='tight')

    elif name == "timeomp":
        for ndim, group_ndim in df.groupby('ndim'):
            plt.figure()
            for groupsize, group_groupsize in group_ndim.groupby('groupsize'):
                x = group_groupsize['nthread']
                y = group_groupsize['timefull_avg']
                plt.plot(x, y, label=f"gs={groupsize}", marker='x')
            size = group_groupsize['size'].iloc[0]
            kernel_type = group_groupsize['kernel_type'].iloc[0]
            interp_order = group_groupsize['interp_order'].iloc[0]
            tree_height = group_groupsize['tree_height'].iloc[0]
            plt.xlabel('Number of threads')
            plt.ylabel('Time (s)')
            plt.title(f'Time vs. Number of threads (d={ndim},s={size},k={kernel_type},o={interp_order},h={tree_height})')
            plt.legend()
            plt.savefig(f'scalfmm_timeomp_d={ndim}.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()