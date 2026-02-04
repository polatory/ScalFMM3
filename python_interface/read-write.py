import argparse
import numpy as np 

from fmmTools import fma_reader, fma_writer


parser = argparse.ArgumentParser(prog="read-write")

parser.add_argument("-if", "--input_file", type=str,  required=True, help="Particles file to read in fma/bfma format")
parser.add_argument("-of", "--output_file",default="output.fma",  type=str, help="Particles file to write in fma format")

# Create the parser

args = parser.parse_args()

filename = args.input_file

#
# Read the particles from file filename
#   particles is an array of size 3 containaing numpy arrays
#    particles[0]  the position matrix of size nb_particles x dimension
#    particles[1]  the input matrix of size nb_particles x nb_input
#    particles[2]  the output matrix of size nb_particles x nb_output
#   centre is a Numpy array containing the centre of the simulation box
#   width is the width of the simulation box
particles, centre, width = fma_reader(filename,verbose=True)

print("centre", centre, type(centre))
print("size", width,  type(width))
filename = args.output_file

# remove z axis
pos = particles[0]
particles[0] = pos[:,0:2]
centre = centre[0:2]

print("xxx centre", centre, type(centre))

#
# Write the new particles 
fma_writer(filename,particles, centre, width)
