import os
import sys
import numpy as np 
from pyfmm import fmaloader, fmawriter, point


def get_real_type(data):

    if data == 4:
        return np.float 
    elif data== 8:
        return np.float64 
    else :
        sys.exit("Oops!  no valid number (4 or 8).")

def fmareader_ascii(filename,verbose=False):
    '''
        Read an FMA file of scalfmmm 

        Return 
            particules an array of three objects (Numpy array) (position, inputs, ou tputs)
            centre of the box
            the size of the box
    '''

    # Read the two first line of the header
    file = open(filename, "r")
    line1 = file.readline()
    lline = line1.split()
    data_type = int(lline[0])
    my_type = get_real_type(data_type)
    nval= int(lline[1])
    dimension = int(lline[2])
    nb_inputs = int(lline[3])
    nb_outputs = nval -  dimension - nb_inputs
    #
    #
    line2 = file.readline() 
    lline = line2.split()
    npoints =  int(lline[0])
    size = my_type(lline[1])*2.0
    centre = np.zeros(dimension, dtype = my_type)
    for i in range(dimension):
        centre[i] = my_type(lline[i+2])
    file.close()
    # Read the particles
    particles = np.empty((3), dtype=object)
    particles[0] = np.loadtxt(filename,  comments='#', skiprows=2, usecols=range(0,dimension))

    particles[1] = np.loadtxt(filename,  comments='#', skiprows=2, usecols=range(dimension,(dimension+nb_inputs))).reshape((npoints,nb_inputs))
    if nb_outputs == 0:
        particles[2] = np.zeros((npoints,1),dtype=my_type) 
    else:
        start = dimension+nb_inputs 
        particles[2] = np.loadtxt(filename,  comments='#', skiprows=2, usecols=range(start,(start+nb_outputs))).reshape(npoints,nb_outputs)

    if (verbose):
        print("   dimension:  ", dimension, "   nb_inputs:  ", nb_inputs,"   nb_outputs: ", nb_outputs)
        print("   centre:     ", centre,"   size:       ", size)

    return particles,centre,size
    
def fmareader_binary(filename,verbose=False):
    '''
        Read an FMA file of scalfmmm 

        Return 
            particules an array of three objects (position, inputs, ou tputs)
            centre of the box
            the size of the box
    '''
    loader = fmaloader(name=filename, verbose=verbose)
    
    my_type = get_real_type(loader.data_type())

    nb_particles = loader.n()
    nb_data = loader.nb_data()
    dim = loader.dim()
    nb_inputs = loader.nb_inputs()
    nb_outputs = loader.nb_outputs()
    data = loader.read()
    particles = np.empty((3), dtype=object)
    particles[0] = data[:,0:dim]
    particles[1] = data[:,dim:dim+nb_inputs]
    if nb_outputs == 0:
        particles[2] = np.zeros((nb_particles,1),dtype=my_type) 
    else:
        start = dim+nb_inputs 
        particles[2] = data[:,start:start+nb_outputs]

    centre_tmp = loader.centre()
    width = my_type(loader.width())
    centre = np.zeros((dim),dtype=my_type)
    for i in range(dim):
        centre[i] = centre_tmp[i]
    return particles,centre,width

def fma_reader(filename,verbose=False):
    '''
        Read an FMA file of scalfmmm 

        Return 
            particules an array of three objects (position, inputs, ou tputs)
            centre of the box
            the size of the box
    '''
    # Chemin du fichier
    print("read file ",filename)
    # Extraction de l'extension
    extension = os.path.splitext(filename)
    if  extension[-1] == '.fma':
        return fmareader_ascii(filename,verbose)
    elif  extension[-1] == '.bfma':
        return fmareader_binary(filename,verbose)
    else:
        sys.exit("Wrong extention. only .fma (ascii) or .bfma (binary) files are avalaible")


def fmawriter_ascii(filename,particles, centre, width,verbose=False):
    '''
        write particles in  FMA file of scalfmmm 

    '''
    print(" -------------- fmawriter_ascii ------------------")
    # write the two first line of the header
    file = open(filename, "w")
    #   DatatypeSize  Number_of_record_per_line dimension Number_of_input_data
    #   NB_particles  half_Box_width  Center (dim values)
    #   Particle_values
    print("fmawriter_ascii ", centre, type(centre))
    p_shape = particles.shape
    print(type(p_shape),p_shape)
    if particles[0].dtype == 'float64' or particles[0].dtype == 'np.float64':
        line1 = '8'
    else:
        line1 = '4'
    n = 0
    for i in range(p_shape[0]):
        n += particles[i].shape[1]
    dimension = particles[0].shape[1]
    line1 += ' ' + str(n) + ' ' + str(dimension) 
    if p_shape[0] > 1:
        nb_inputs = particles[1].shape[1]
        line1 += ' ' + str(nb_inputs) #nb_inputs
    else:
        nb_inputs = 0 
        line1 += ' 0 '
    file.write(line1+'\n')
    nb_outputs = n -  dimension - nb_inputs
    #
    nb_part = particles[0].shape[0]
    line2 = str(nb_part)
    line2 += ' ' + str(width/2 )
    for i in range(dimension):
        line2 += ' ' + str(centre[i])
    file.write(line2+'\n')
    pos = particles[0]
    for p in range(nb_part):
        linep = ''
        for i in range(p_shape[0]):
            current = particles[i]
            for j in range(current.shape[1]):
                linep += str(current[p, j]) + ' '

        file.write(linep+'\n')
    file.close()

    #
def fmawriter_binary(filename,particles, centre, width,verbose=False):
    '''
        write particles in  FMA file of scalfmmm 

    '''
    writer = fmawriter(name=filename, binary=True)
    n = 0
    for i in range(particles.shape[0]):
        print(i,particles[i].dtype)
        n += particles[i].shape[1]
    n_particles =  particles[0].shape[0]
    array = np.zeros([n_particles, n], dtype=particles[0].dtype, order='C')

    start = 0 
    for i in range(particles.shape[0]):
        end = start + particles[i].shape[1]
        array[:,start:end] = particles[i][:,0:particles[i].shape[1]]
        start = end

    if particles[0].dtype == 'float64' or particles[0].dtype == np.float64:
        dataType = 8
    else:
        dataType = 4
 
    writer.write_header(centre, width, n_particles, dataType,dataType, particles[0].shape[1], particles[1].shape[1])

    writer.write(array, n, n_particles)
    writer.close()


#
def fma_writer(filename, particles, centre, width, verbose=False):
    '''
        write  an FMA file of scalfmmm 
        Inputs
            particules an array of three objects (position, inputs, outputs)
            centre of the box
            the size of the box
    '''
    # Chemin du fichier
    print("write file ",filename)
    # Extraction de l'extension
    extension = os.path.splitext(filename)
    if  extension[-1] == '.fma':
        fmawriter_ascii(filename,particles, centre, width,verbose)
    elif  extension[-1] == '.bfma':
        fmawriter_binary(filename,particles, centre, width,verbose)

    else:
        sys.exit("Wrong extention. only .fma (ascii) files are avalaible")


########################################################
if __name__ == '__main__':
    filename = "../data/cubeSorted_10_PF.fma"

    particles, centre, size = fma_reader(filename)

    print("centre", centre)
    print("size", size)

    print(particles)
    fmawriter_ascii('toto.fma',particles, centre, size)