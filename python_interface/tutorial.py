import random as rd
import numpy as np
import argparse
import time
import sys
from colorama import Fore, Back, Style
from argparse import Namespace

import fmmTools 
import pyfmm  # make sure that the module is available

# - Do not forget to update the PYTHONPATH environment variable as follows:
#       export SCALFMM_BUILD=/SCALPFMM_BUILD_PATH/
#       export PYTHONPATH=${SCALFMM_BUILD}/python_interface:$PYTHONPATH
#
# - Run python script:
#       python tutorial.py --help
#       python tutorial.py
# python ../python_interface/tutorial.py --group-size 1 --order 5 --tree-height 3 --number-particles 1000 --center 0 0 --width 2 --input_file  ../data/datal2p_10.fma
# python ../python_interface/tutorial.py --group-size 1 --order 5 --tree-height 3 --number-particles 1000 --center 0 0 --width 2
def run(args):
    """Run ScalFMM tutorial"""


    dimension = 2
    _order = args.order
    _group_size = args.group_size
    _tree_height = args.tree_height

    _omp = args.open_mp
    _mutual = args.mutual

    _center = args.center
    _width = args.width
    _file = None
    if args.input_file != None:
        _file = args.input_file
    else:
        _center = args.center
        _width = args.width   
        _nb_particles = args.number_particles

    if _file:
        data,_center,_width = fmmTools.fmareader(_file,verbose=True)
        particles = data[0]
        inputs = data[1]
        outputs = data[2]
        _nb_particles = particles.shape[0]
    else:
        # Init random seed
        rd.seed(123)
        data = np.empty((3), dtype=object)
        data[0] = np.zeros((_nb_particles, dimension), dtype=np.float64)
        data[1] = np.zeros((_nb_particles, 1), dtype=np.float64,)
        data[2] = np.zeros((_nb_particles, 1), dtype=np.float64)
        # particles in box
        for i in range(dimension):
            lower_bd = _center[i] - 0.5 * _width
            upper_bd = _center[i] + 0.5 * _width
        particles =  data[0] 
        for i in range(_nb_particles):
            particles[i,:] = rd.uniform(lower_bd, upper_bd)
            data[1][i] = rd.uniform(1, dimension)
            inputs = data[1]
            outputs = data[2]

    if not (dimension == particles.shape[1]):
        sys.exit("Dimension is not 2 ")
    # ----------------------
    # Initialization
    # -----------------------

    print(Fore.LIGHTBLACK_EX + "\n\t--- initialization ---" + Style.RESET_ALL)
    center = pyfmm.point(_center[0], _center[1])
    box = pyfmm.box(width=_width, box_center=center)

    print("\tsimulation box =", end="")
    print(box)
    print("\tcenter = ", end="")
    print(center)

    level = _tree_height -1 


    # construct the Morton index of the particles. The vector is sorted
    #  and we have the permutation to apply to the data
    morton, perm = pyfmm.morton_sort(box, particles ,level )  
    #  Apply the permutation
    for i in range(particles.shape[1]):
        particles[:,i] = particles[perm[:], i ]   
    for i in range(inputs.shape[1]):
        inputs[:,i] = inputs[perm[:], i ]
    for i in range(outputs.shape[1]):
        outputs[:,i] = outputs[perm[:], i ]   

    # Tree
    # -----------------------
    print(Fore.LIGHTBLACK_EX + "\n\t--- tree ---" + Style.RESET_ALL)
    t = time.time()
    # build and empty tree  
    tree = pyfmm.tree(
        tree_height=_tree_height,
        order=_order,
        group_size_leaves=_group_size,
        group_size_cells=_group_size,
        box=box
    )
    # set the tree structure (leaves/cells)
    tree.build_with_morton( morton )
    # fill particles position in the leaves 
    tree.fill_particles(particles)
    # fill in the positions of the particles in the leaves 
    tree.set_inputs(inputs)      
    t_tree = time.time() - t
    print(f"\ttime required: {t_tree:.7} s")

    # ---------------------------------------------
    #                FMM computation
    # ---------------------------------------------

    # ---------------------------------------------
    #      build FMM operator
    # ---------------------------------------------

    print(Fore.LIGHTBLACK_EX + "\n\t--- fmm preprocessing ---" + Style.RESET_ALL)
    t = time.time()
    interpolator = pyfmm.interpolator(
        order=_order, tree_height=_tree_height, root_cell_width=box.width(0)
    )
    far_field = pyfmm.far_field(interpolator=interpolator)
    near_field = pyfmm.near_field()
    near_field.mutual = _mutual
    fmm_operator = pyfmm.fmm_operator(near_field=near_field, far_field=far_field)
    t_preproc = time.time() - t
    print(f"\ttime required: {t_preproc:.7} s")

    # ---------------------------------------------
    #         computation
    # ---------------------------------------------

    print(Fore.LIGHTBLACK_EX + "\n\t--- fmm computation ---" + Style.RESET_ALL)
    if _omp:
        print("\tusing parallel version (OpenMP)...")
        t = time.time()
        pyfmm.fmm_omp(tree=tree, fmm_operators=fmm_operator)
        t_fmm = time.time() - t
        print(f"\ttime required: {t_fmm:.7} s")
    else:
        print("\tusing sequential version...")
        t = time.time()
        pyfmm.fmm_seq(tree=tree, fmm_operators=fmm_operator)
        t_fmm = time.time() - t
        print(f"\ttime required: {t_fmm:.7} s")


    # --------------------------------------------
    #             Direct computation
    # ---------------------------------------------

    # ----------------------
    # Particle container for direct computation
    # -----------------------

    print(Fore.LIGHTBLACK_EX + "\n\t--- particle container ---" + Style.RESET_ALL)
    t = time.time()
    container = pyfmm.container(_nb_particles)
    idx = 0
    for part in container:
        for i in range(2):
            part.set_position(i, particles[idx,i])
        part.set_input(0, inputs[idx,0])
        part.set_output(0, outputs[idx,0])
        part.set_variable(idx)
        idx += 1
    t_container = time.time() - t
    print(f"\ttime required: {t_container:.7} s")
    # ----------------------
    print(Fore.LIGHTBLACK_EX + "\n\t--- direct computation ---" + Style.RESET_ALL)
    mk = pyfmm.matrix_kernel()
    t = time.time()
    pyfmm.full_direct(container, mk)
    t_direct = time.time() - t
    print(f"\ttime required: {t_direct:.7} s")

    # ----------------------
    # Postprocessing
    # -----------------------

    print(Fore.LIGHTBLACK_EX + "\n\t--- postprocessing ---" + Style.RESET_ALL)

    out_direct = np.zeros((_nb_particles))
    out_fmm = np.zeros((_nb_particles))

    container.get_outputs(out_direct)
    out_direct = out_direct.reshape(outputs.shape)

    tree.get_outputs(outputs)
    l2_error, infty_error, _ = pyfmm.compute_error(out_direct=out_direct, out_fmm=outputs)

    speedup = t_direct / t_fmm

    print(Fore.GREEN + f"\t - speed-up                = {speedup:.7}" + Style.RESET_ALL)
    print(Fore.GREEN + f"\t - L2 relative error       = {l2_error:.7}" + Style.RESET_ALL)
    print(Fore.GREEN + f"\t - infinity relative error = {infty_error:.7}" + Style.RESET_ALL)

    # save data from tree in file ( fma (ascii) format or (bfma) binary)
    writer = pyfmm.fmawriter(name="toto.bfma",binary=True)
    writer.write(tree,_nb_particles)

def parse_arguments():
    """Parse command-line arguments."""

    # Create the parser
    parser = argparse.ArgumentParser(description="Scalfmm tutorial")

    parser.add_argument("--open-mp","-omp", action="store_true", required=False, help="Use parallel version of the fmm algorithm")
    parser.add_argument("--mutual", "-m", action="store_true", required=False, help="Enable full mutual computation for the p2p operator")
    parser.add_argument("-gs", "--group-size", type=int, required=True, help="Group tree chunk size (granularity)", default=1)
    parser.add_argument("-th", "--tree-height", type=int, required=True, help="Tree height (or initial height in case of an adaptive tree)", default=3)
    parser.add_argument("-o", "--order", type=int, required=True, help="Precision setting (order of the interpolation method)", default=5)
    parser.add_argument("-N", "--number-particles", type=int,  help="Number of particles", default=1000)
    parser.add_argument("-c", "--center", type=float, nargs="+",  help="Center of the simulation box", default=[0.,0.])
    parser.add_argument("-w", "--width", type=float,  help="Width of the simulation box", default=2.)
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity level.")
    parser.add_argument("-fi", "--input_file", type=str, help="Particles file in fma format")
    parser.add_argument("--random",  action='store_true',  required=False, help="use random particles then --center and -- width are required.")

    args = parser.parse_args()
    if args.random  and not args.center and not args.width:
        parser.error("Args --random required  args:  --number-particles --centre and --width")
    if args.random  and args.input_file:
        parser.error("Use either --random or --input_file")
    return args

def print_args(args):
    """Print arguments form parser."""
    print(Fore.CYAN + "\n\t--- parameters ---" + Style.RESET_ALL)
    for arg, value in vars(args).items():
        print(Fore.CYAN + f"<params> {arg} = {value}" + Style.RESET_ALL)

def main():
    """Main function."""
    args = parse_arguments()
    print_args(args) if args.verbose >= 2 else None


    run(args)

if __name__ == "__main__":
    main()
