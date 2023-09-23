// --------------------------------
// See LICENCE file at project root
// File : tools/fma_loader.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_FMA_LOADER_HPP
#define SCALFMM_TOOLS_FMA_LOADER_HPP

#include <array>
#include <cstdlib>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <string.h>
#include <string>
#include <vector>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/for_each.hpp"

namespace scalfmm::io
{
    /**\class FFmaGenericLoader
     * @author Olivier Coulaud (olivier coulaud@inria.fr)
     * \warning This class only works in shared memory (doesn't work with MPI).
     *
     * \brief Reads an FMA formatted particle file.
     *
     * The file may be in ascii or binary mode.  There are several overloads of the
     * fillParticle(FPoint<FReal>*, FReal*) member function to read data from a file. The
     * example below shows how to use the loader to read from a file.
     *
     *
     * \code
     * // Instanciate the loader with the particle file.
     * FFmaGenericLoader<FReal> loader("../Data/unitCubeXYZQ20k.fma"); // extension fma -> ascii format
     * // Retrieve the number of particles
     * std::size_t nbParticles = loader.getNumberOfParticles();
     *
     * // Create an array of particles, initialize to 0.
     * FmaRParticle * const particles = new FmaRParticle[nbParticles];
     * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
     *
     * // Read the file via the loader.
     * for(std::size_t idx = 0 ; idx < nbParticles ; ++idx){
     *     loader.fillParticle(particles[idx]);
     * }
     * \endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * \code
     *   DatatypeSize  Number_of_record_per_line dimension Number_of_input_data
     *   NB_particles  half_Box_width  Center (dim values)
     *   Particle_values
     * \endcode
     *
     * `DatatypeSize` can have one of two values:
     *  - 4, float ;
     *  - 8, double.
     *
     * `Number_of_records_per_line` gives the data count for each line of
     * the `Particle_values`. For example :
     *  - 4, the particle values are X Y Z Q;
     *  - 8, the particle values are X Y Z Q  P FX FY FZ<br>
     *
     */
    template<class FReal, int Dimension = 3>
    class FFmaGenericLoader
    {
      protected:
        std::fstream* m_file;                               ///< the stream used to read the file
        bool m_binaryFile;                                  ///< if true the file to read is in binary mode
        container::point<FReal, Dimension> m_centerOfBox;   ///< The center of box (read from file)
        std::vector<FReal> m_center{};                      ///< The center of box (read from file)
        FReal m_boxWidth;                                   ///< the box width (read from file)
        std::size_t m_nbParticles;                          ///< the number of particles (read from file)
        std::array<unsigned int, 4> m_typeData;   ///< Size of the data to read, number of data on 1 line, dimension
                                                  ///< of space and number of input values
        std::string m_filename;                   ///< file name containung the data
        bool m_verbose;                           ///<  Verbose mode

      private:
        FReal* m_tmpVal;   ///< Temporary array to read data
        /// Count of other data pieces to read in a particle record after the 4 first ones.
        unsigned int m_otherDataToRead;

        void open_file(const std::string filename, const bool binary, const bool verbose = true)
        {
            m_filename = filename;
            m_verbose = verbose;
            if(binary)
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::in | std::ios::binary);
            }
            else
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::in);
            }
            // test if open
            if(!this->m_file->is_open())
            {
                std::cerr << "File " << filename << " not opened! Error: " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(m_verbose)
            {
                std::cout << "FFmaGenericLoader file " << filename << " opened" << std::endl;
            }
        }

      public:
        using dataType = FReal;

        /**
         * This constructor opens a file and reads its header. The file will be
         * kept opened until destruction of the object.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         * - All information accessible in the header can be retreived after this call.
         * - To test if the file has successfully been opened, call hasNotFinished().
         *
         * @param filename the name of the file to open. Must end with `.fma` or `.bfma`.
         */
        FFmaGenericLoader(const std::string& filename, bool verbose, bool old_format = false)
          : m_file(nullptr)
          , m_binaryFile(false)
          , m_centerOfBox{0.0}
          , m_boxWidth(0.0)
          , m_nbParticles(0)
          , m_filename(filename)
          , m_verbose(verbose)
          , m_tmpVal(nullptr)
          , m_otherDataToRead(0)
        {
            // open particle file
            if(filename.find(".bfma") != std::string::npos)
            {
                m_binaryFile = true;
            }
            else if(filename.find(".fma") != std::string::npos)
            {
                m_binaryFile = false;
            }
            else
            {
                std::cout << "FFmaGenericLoader: "
                          << "Only .fma or .bfma input file are allowed. Got " << filename << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            this->open_file(filename, m_binaryFile, m_verbose);
            this->readHeader(old_format);
        }
        void close()
        {
            m_file->close();
            delete m_file;
            m_file = nullptr;
        }
        /**
         * Default destructor, closes the file
         */
        virtual ~FFmaGenericLoader()
        {
            if(m_file != nullptr)
            {
                m_file->close();
            }
            delete[] m_tmpVal;
        }

        /**
         * To know if file is open and ready to read
         * @return true if loader can work
         */
        bool isOpen() const { return this->m_file->is_open() && !this->m_file->eof(); }

        /**
         * To get the number of particles from this loader
         */
        std::size_t getNumberOfParticles() const { return this->getParticleCount(); }

        /**
         * The center of the box from the simulation file opened by the loader
         * @return box center
         */
        auto /*container::point<FReal, Dimension> */ getCenterOfBox() const -> container::point<FReal, Dimension>
        {
            return this->getBoxCenter();
        } /**
           * The center of the box from the simulation file opened by the loader
           * @return box center
           */
        ///
        /// \brief getPointerCenterOfBox return a pointer on the element of the Box center
        ///
        auto getPointerCenterOfBox() const -> FReal* { return this->m_center.data(); }

        /**
         * \brief Get the distribution particle count
         * \return The distribution particle count
         */
        std::size_t getParticleCount() const { return this->m_nbParticles; }

        /**
         * \brief Get the center of the box contining the particles
         *
         * \return A point (ontainer::point<FReal>) representing the box center
         */
        auto getBoxCenter() const { return this->m_centerOfBox; }

        /**
         * The box width from the simulation file opened by the loader
         * @return box width
         */
        FReal getBoxWidth() const { return this->m_boxWidth; }
        /**
         * The box width from the simulation file opened by the loader
         * @return the number of data per record (Particle)
         */
        unsigned int getNbRecordPerline() { return m_typeData[1]; }
        /**
         * The box width from the simulation file opened by the loader
         * @return the Dimension space
         */
        unsigned int get_dimension() { return m_typeData[2]; }
        /**
         * The box width from the simulation file opened by the loader
         * @return the number of input data per record (Particle)
         */
        unsigned int get_number_of_input_per_record() { return m_typeData[3]; }
        /**
         * The box width from the simulation file opened by the loader
         * @return the number of ioutput data per record (Particle)
         */
        unsigned int get_number_of_output_per_record() { return m_typeData[1] - m_typeData[2] - m_typeData[3]; }
        /**
         * To know if the data are in float or in double type
         * @return the type of the values float (4) or double (8)
         */
        unsigned int getDataType() { return m_typeData[0]; }

        auto get_header_size()
        {
            return m_typeData.size() * sizeof(unsigned int) + sizeof(std::size_t) + sizeof(m_boxWidth) +
                   sizeof(FReal) * m_typeData[2];
        }

        /**
         * \brief Fill a particle set from the current position in the file.
         *
         * @param dataToRead   array of type FReal. It contains all the values of a
         * particles (for instance X,Y,Z,Q, ..)
         *
         * @param nbDataToRead number of value to read (I.e. size of the array)
         */
        void fillParticle(FReal* dataToRead, const unsigned int nbDataToRead)
        {
            if(m_binaryFile)
            {
                m_file->read((char*)(dataToRead), sizeof(FReal) * nbDataToRead);
                if(nbDataToRead < m_typeData[1])
                {
                    m_file->read((char*)(this->m_tmpVal), sizeof(FReal) * (m_typeData[1] - nbDataToRead));
                }
            }
            else
            {
                //           std::cout << " read " << nbDataToRead << " of " << typeData[1] << "  ";
                for(unsigned int i = 0; i < nbDataToRead; ++i)
                {
                    (*this->m_file) >> dataToRead[i];
                    //                   std::cout << dataToRead[i] << " ";
                }
                //               std::cout << '\n';
                if(nbDataToRead < m_typeData[1])   // skip extra data
                {
                    FReal x;
                    for(unsigned int i = 0; i < m_typeData[1] - nbDataToRead; ++i)
                    {
                        (*this->m_file) >> x;
                    }
                }
            }
        }

        /**
         * Fill a set of particles form the current position in the file.
         *
         * If the file is a binary file and we read all record per particle then we
         * read and fill the array in one instruction.
         *
         * @tparam dataPart  the particle class. It must implement the members
         * getPtrFirstData(), getReadDataNumber(). See FmaRWParticle.
         *
         * @param dataToRead the array of particles to fill.
         * @param N          the number of particles.
         */
        template<class vector_type>
        void fillParticles(vector_type& dataToRead, const std::size_t N)
        {
            if(dataToRead.size() != m_typeData[1] * N)
            {
                std::cerr << "Error in fFFmaGenericLoader::fillParticle(dataPart *dataToRead, const  std::size_t N)."
                          << std::endl
                          << "Wrong number of values to read:" << std::endl
                          << "expected " << m_typeData[1] << " from file\n"
                          << "expected " << dataToRead.size() / N << " from structure." << std::endl;
                std::cerr << "Read from file: " << this->m_filename << std::endl;
                throw(" ToDo\n ");
                std::exit(EXIT_FAILURE);
            }

            if(m_binaryFile)
            {
                m_file->read((char*)(dataToRead.data()), sizeof(FReal) * m_typeData[1] * N);
            }
            else
            {
                for(std::size_t i = 0; i < dataToRead.size(); i += m_typeData[1])
                {
                    this->fillParticle(&(dataToRead[i]), m_typeData[1]);
                }
            }
        }

      private:
        void readHeader(const bool old_format = true)
        {
            int nbval = 4;
            if(old_format)
            {
                nbval = 2;
            }
            if(this->m_binaryFile)
            {
                this->readBinaryHeader(nbval);
            }
            else
            {
                this->readAscciHeader(nbval);
            }
            if(Dimension < m_typeData[2])
            {
                std::cerr << "Wrong dimension. Template parameter is " << Dimension << " and dimension in file is "
                          << m_typeData[2] << ".\n";
            }
            // if(m_typeData[1] != sizeof(FReal)) {
            //     std::string type[2]{"float","double"};
            //     std::cout << type[0] << " "<< type[1]<< m_typeData[0] <<  " "<< m_typeData[0] <<  " "<< m_typeData[1]
            //     <<std::endl; std::cout << "Data conversion is needed (Reader in "<< type[sizeof(FReal)/4-1]
            //     << ") data in file are in " << type[m_typeData[0]/4-1]<< std::endl;
            // }
            if(m_verbose)
            {
                std::cout << "   nbParticles: " << this->m_nbParticles << std::endl
                          << "   Box width:   " << this->m_boxWidth << std::endl
                          << "   Center:       [ ";
                for(unsigned int i = 0; i < m_typeData[2] - 1; ++i)
                {
                    std::cout << this->m_centerOfBox[i] << ", ";
                }
                std::cout << this->m_centerOfBox[m_typeData[2] - 1] << " ]" << std::endl;
            }
        }
        void readAscciHeader(const int nbVal)
        {
            if(m_verbose)
            {
                std::cout << " File open in ASCII mode " << nbVal << std::endl;
                std::cout << "   Datatype ";
            }
            FReal x;
            for(int i = 0; i < nbVal; ++i)
            {
                (*this->m_file) >> m_typeData[i];
            }
            if(nbVal == 2)   // Old Format
            {
                m_typeData[2] = 3;
                m_typeData[3] = 1;
            }
            (*this->m_file) >> this->m_nbParticles >> this->m_boxWidth;
            if(m_verbose)
            {
                std::cout << this->m_nbParticles << "  " << this->m_boxWidth << '\n';
            }
            m_center.resize(m_typeData[2]);
            for(unsigned int i = 0; i < m_typeData[2]; ++i)
            {
                (*this->m_file) >> x;
                this->m_centerOfBox[i] = x;
                this->m_center[i] = x;
            }
            this->m_boxWidth *= 2;

            m_otherDataToRead = m_typeData[1] - m_typeData[2] - m_typeData[3];   // output variables
        };
        void readBinaryHeader(const int nbVal)
        {
            if(m_verbose)
            {
                std::cout << " File open in binary mode " << std::endl;
            }
            m_file->seekg(std::ios::beg);
            m_file->read((char*)&m_typeData, nbVal * sizeof(unsigned int));
            if(nbVal == 2)   // Old Format
            {
                m_typeData[2] = 3;
                m_typeData[3] = 1;
            }
            if(m_typeData[0] != sizeof(FReal))
            {
                std::cerr << "Size of elements in part file " << m_typeData[0] << " is different from size of FReal "
                          << sizeof(FReal) << std::endl;
                std::cerr << "The conversion is not yet implemented for binary file\n";
                std::exit(EXIT_FAILURE);
            }
            else
            {
                m_file->read((char*)&(this->m_nbParticles), sizeof(std::size_t));
                m_file->read((char*)&(this->m_boxWidth), sizeof(this->m_boxWidth));
                this->m_boxWidth *= 2;

                FReal* x = new FReal[m_typeData[2]];
                m_file->read((char*)x, sizeof(FReal) * m_typeData[2]);
                if(m_typeData[2] != Dimension)
                {
                    std::cerr << "m_typeData[2] != Dimension: " << m_typeData[2] << " != " << Dimension << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                // std::cout << " " << m_typeData[2] << "  centre: ";

                for(unsigned int i = 0; i < m_typeData[2]; ++i)
                {
                    this->m_centerOfBox[i] = x[i];
                }
                // std::cout << this->m_centerOfBox << std::endl;
            }
            m_otherDataToRead = m_typeData[1] - m_typeData[2] - m_typeData[3];   // output variables
            if(m_otherDataToRead > 0)
            {
                m_tmpVal = new FReal[m_otherDataToRead];
            }
        }
    };

    /**
     * \warning This class only works in shared memory (doesn't work with MPI).
     *
     * \brief Writes a set of particles to an FMA formated file.
     *
     * The file may be in ASCII or binary mode. The example below shows how to use the class.
     *
     * \code
     * // Instantiate the writer with a binary fma file (extension .bfma).
     * FFmaGenericWriter<FReal> writer ("data.bfma");
     *
     * // Write the header of the file.
     * writer.writeHeader(loader.getCenterOfBox(), loader.getBoxWidth(), NbPoints, sizeof(FReal), nbData);
     *
     * // Write the data. Here particles is an array and a particle has nbData values.
     * writer.writeArrayOfReal(particles, nbData, NbPoints);
     * \endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * \code
     *   DatatypeSize  Number_of_record_per_line dimension Number_of_input_data
     *   NB_particles  half_Box_width  Center (dim values)
     *   Particle_values
     * \endcode
     *
     * `DatatypeSize` can have one of two values:
     *  - 4, float;
     *  - 8, double.
     *
     * `Number_of_records_per_line` gives the data count for each line of
     * the `Particle_values`. For example :
     *  - 4, the particle values are `X Y Z Q`;
     *  - 8, the particle values are `X Y Z Q  P FX FY FZ`.
     */
    template<class FReal>
    class FFmaGenericWriter
    {
      protected:
        std::fstream* m_file;   ///< the stream used to write the file
        bool m_binaryFile;      ///< if true the file is in binary mode

      public:
        /**
         * This constructor opens a file to be written to.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         *
         * @param filename the name of the file to open.
         */
        FFmaGenericWriter(const std::string& filename)
          : m_binaryFile(false)
        {
            std::cout << "FFmaGenericWriter filename " << filename << std::endl;
            std::string ext(".bfma");
            // open particle file
            if(filename.find(".bfma") != std::string::npos)
            {
                m_binaryFile = true;
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out | std::ios::binary);
            }
            else if(filename.find(".fma") != std::string::npos)
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out);
                this->m_file->precision(std::numeric_limits<FReal>::digits10);
            }
            else
            {
                std::cout << "filename " << filename << " find fma " << filename.find(".fma") << " "
                          << std::string::npos << std::endl;
                std::cout << "Output file not allowed only .fma or .bfma extensions" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            // test if open
            if(!this->m_file->is_open())
            {
                std::cerr << "File " << filename << " not opened! " << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "FFmaGenericWriter file " << filename << " opened" << std::endl;
        }

        /**
         * This constructor opens a file to be written to.
         *
         * @param filename the name of the file to open.
         * @param binary   true if the file to open is in binary mode
         */
        FFmaGenericWriter(const std::string& filename, const bool binary)
          : m_file(nullptr)
          , m_binaryFile(binary)
        {
            if(binary)
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out | std::ios::binary);
            }
            else
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out);
                this->m_file->precision(std::numeric_limits<FReal>::digits10);
            }
            // test if open
            if(!this->m_file->is_open())
            {
                std::cerr << "File " << filename << " not opened! " << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "FFmaGenericWriter file " << filename << " opened" << std::endl;
        }
        void close()
        {
            m_file->close();
            delete m_file;
            m_file = nullptr;
        }
        /**
         * Default destructor, closes the file.
         */
        virtual ~FFmaGenericWriter()
        {
            if(m_file != nullptr)
            {
                m_file->close();
            }
        }

        /**
         * To know if file is open and ready to read
         * @return true if loader can work
         */
        bool isOpen() const { return this->m_file->is_open() && !this->m_file->eof(); }

        /**
         * To know if opened file is in binary mode
         * @return true if opened file is in binary mode
         */
        bool isBinary() const { return this->m_binaryFile; }

        /**
         * Writes the header of FMA file.
         *
         * Should be used if we write the particles with writeArrayOfReal method
         *
         * @param centerOfBox      The center of the Box (FPoint<FReal> class)
         * @param boxWidth         The width of the box
         * @param nbParticles      Number of particles in the box (or to save)
         * @param dataType         Size of the data type of the values in particle
         * @param nbDataPerRecord  Number of record/value per particle
         */
        template<class POINT_T>
        void writeHeader(const POINT_T& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                         const unsigned int dataType, const unsigned int nbDataPerRecord, const unsigned int dimension,
                         const unsigned int nb_input_values)
        {
            std::array<unsigned int, 4> typeFReal = {dataType, nbDataPerRecord, dimension, nb_input_values};
            FReal x = boxWidth * FReal(0.5);
            if(this->m_binaryFile)
            {
                this->writerBinaryHeader(centerOfBox, x, nbParticles, typeFReal.data(), 4);
            }
            else
            {
                this->writerAscciHeader(centerOfBox, x, nbParticles, typeFReal.data(), 4);
            }
            std::cout << "   Datatype ";
            for(int i = 0; i < 4; ++i)
            {
                std::cout << " " << typeFReal[i];
            }
            std::cout << '\n';
            std::cout << "   nbParticles: " << nbParticles << std::endl
                      << "   Box width:   " << boxWidth << std::endl
                      << "   Center:      ";
            for(auto e: centerOfBox)
            {
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }
        template<class POINT_T>
        void writeHeaderOld(const POINT_T& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                            const unsigned int dataType, const unsigned int nbDataPerRecord,
                            const unsigned int dimension, const unsigned int nb_input_values)
        {
            std::array<unsigned int, 4> typeFReal = {dataType, nbDataPerRecord, dimension, nb_input_values};
            FReal x = boxWidth * FReal(0.5);
            if(this->m_binaryFile)
            {
                this->writerBinaryHeader(centerOfBox, x, nbParticles, typeFReal.data(), 2);
            }
            else
            {
                this->writerAscciHeader(centerOfBox, x, nbParticles, typeFReal.data(), 2);
            }
        }

        /**
         *  Write an array of data in a file Fill
         *
         * @param dataToWrite array of particles of type FReal
         * @param nbData number of data per particle
         * @param N number of particles
         *
         *   The size of the array is N*nbData
         *
         *   example
         * \code
         * FmaRParticle * const particles = new FmaRParticle[nbParticles];
         * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
         * ...
         * FFmaGenericWriter<FReal> writer(filenameOut) ;
         * Fwriter.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
         * Fwriter.writeArrayOfReal(particles, nbParticles);
         * \endcode
         */
        void writeArrayOfReal(const FReal* dataToWrite, const int nbData, const std::size_t N)
        {
            if(m_binaryFile)
            {
                m_file->write((const char*)(dataToWrite), N * nbData * sizeof(FReal));
            }
            else
            {
                this->m_file->precision(std::numeric_limits<FReal>::digits10);

                std::size_t k = 0;
                for(std::size_t i = 0; i < N; ++i)
                {
                    // std::cout << "i "<< i << "  ";
                    for(int jj = 0; jj < nbData; ++jj, ++k)
                    {
                        (*this->m_file) << dataToWrite[k] << "    ";
                        // std::cout      << dataToWrite[k]<< "  ";
                    }
                    (*this->m_file) << std::endl;
                    // std::cout <<std::endl;
                }
                // std::cout << "END"<<std::endl;
            }
        }

        /**
         *  Write all particles (position, input values, output values) inside the octree in fma format into a file.
         *
         * @param[in] tree Octree that contains the particles in the leaves
         * @param[in] number_particles number of particles
         *
         *   example
         * \code
         *  group_tree_type tree(TreeHeight, SubTreeHeight, BoxWidth, CenterOfBox);
         * ...
         * FFmaGenericWriter<FReal> writer(filenameOut) ;
         * Fwriter.writeDataFromOctree(&tree, nbParticles);
         * \endcode
         */
        template<class TREE_T>
        void writeDataFromTree(const TREE_T& tree, const std::size_t& number_particles)
        {
            constexpr int dimension = TREE_T::leaf_type::dimension;
            constexpr int nb_input_elements = TREE_T::leaf_type::particle_type::inputs_size;
            constexpr int nb_output_elements = TREE_T::leaf_type::particle_type::outputs_size;
            constexpr int nb_elt_per_par = dimension + nb_input_elements + nb_output_elements;
            std::cout << "Dimension: " << dimension << std::endl;
            std::cout << "Number of input values: " << nb_input_elements << std::endl;
            std::cout << "Number of output values: " << nb_output_elements << std::endl;
            std::cout << "nb_elt_per_par " << nb_elt_per_par << std::endl;
            //
            using value_type = typename TREE_T::leaf_type::value_type;
            using particles_t = std::array<value_type, nb_elt_per_par>;
            std::vector<particles_t> particles(number_particles);
            //
            int pos = 0;
            scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                              [&pos, &particles](auto& leaf)
                                              {
                                                  for(auto const& it_p: leaf)
                                                  {
                                                      auto& particles_elem = particles[pos++];
                                                      const auto& p = typename TREE_T::leaf_type::particle_type(it_p);
                                                      //
                                                      int i = 0;
                                                      const auto points = p.position();
                                                      for(int k = 0; k < dimension; ++k, ++i)
                                                      {
                                                          particles_elem[i] = points[k];
                                                      }
                                                      // get inputs
                                                      for(int k = 0; k < nb_input_elements; ++k, ++i)
                                                      {
                                                          particles_elem[i] = p.inputs(k);
                                                      }
                                                      // get outputs
                                                      for(int k = 0; k < nb_output_elements; ++k, ++i)
                                                      {
                                                          particles_elem[i] = p.outputs(k);
                                                      }
                                                  }
                                              });
            //
            // write the particles
            const auto& centre = tree.box_center();
            this->writeHeader(centre, tree.box_width(), number_particles, sizeof(value_type), nb_elt_per_par,
                              centre.dimension, nb_input_elements);
            this->writeArrayOfReal(particles.data()->data(), nb_elt_per_par, number_particles);
        }
        ///
        ///  writeDataFrom write all data from the container in fma format
        ///
        ///  How to get automatically double from container
        template<class CONTAINER_T, class POINT_T, typename VALUE_T>
        void writeDataFrom(CONTAINER_T& values, const int& number_particles, const POINT_T& center,
                           const VALUE_T box_width)
        {
            // get the number of elements per particles in the container build with tuples.
            using particle_type = typename CONTAINER_T::value_type;
            constexpr int nb_elt_per_par = meta::tuple_size_v<particle_type>;
            // Not good output_values are put in input_values
            constexpr int nb_input_per_par = particle_type::inputs_size;
            ///  @todo check for different input and output types (double versus complexe)
            using data_type = typename particle_type::outputs_value_type;
            //
            using particles_t = std::array<data_type, nb_elt_per_par>;
            std::vector<particles_t> particles(number_particles);

#pragma omp parallel for shared(particles)
            for(auto it_p = std::begin(values); it_p < std::end(values); ++it_p)
            {
                particles_t particles_elem{};
                int pos = std::distance(std::begin(values), it_p);
                // fill the curent particles inputs outputs in the vector

                scalfmm::meta::for_each(particles_elem, *it_p, [](const auto& tuple_elm) { return tuple_elm; });

                particles[pos] = particles_elem;
            };
            //  write the particles
            // Here we need to separate input from output variables - no tools yet
            this->writeHeader(center, box_width, number_particles, sizeof(data_type), nb_elt_per_par, center.dimension,
                              nb_input_per_par);
            this->writeArrayOfReal(particles.data()->data(), nb_elt_per_par, number_particles);
        }

      protected:
        template<class POINT_T>
        void writerAscciHeader(const POINT_T& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                               const unsigned int* typeFReal, const unsigned int nbVal)
        {
            this->m_file->precision(std::numeric_limits<FReal>::digits10);
            // Line 1
            (*this->m_file) << typeFReal[0];
            for(unsigned int i = 1; i < nbVal; ++i)
            {
                (*this->m_file) << "  " << typeFReal[i];
            }
            (*this->m_file) << '\n';
            // Line 2
            (*this->m_file) << nbParticles << "   " << boxWidth;
            for(std::size_t i = 0; i < centerOfBox.size(); ++i)
            {
                (*this->m_file) << "  " << centerOfBox[i];
                ;
            }

            (*this->m_file) << '\n';
        }
        template<class POINT_T>
        void writerBinaryHeader(const POINT_T& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                                const unsigned int* typeFReal, const unsigned int nbVal)
        {
            m_file->seekg(std::ios::beg);
            m_file->write((const char*)typeFReal, nbVal * sizeof(unsigned int));
            if(typeFReal[0] != sizeof(FReal))
            {
                std::cout << "Size of elements in part file " << typeFReal[0] << " is different from size of FReal "
                          << sizeof(FReal) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else
            {
                m_file->write((const char*)&(nbParticles), sizeof(std::size_t));
                // std::cout << "nbParticles "<< nbParticles<<std::endl;
                m_file->write((const char*)&(boxWidth), sizeof(boxWidth));
                if(nbVal == 2)   // old Format
                {
                    m_file->write((const char*)(&centerOfBox[0]), sizeof(FReal) * 3);
                }
                else
                {
                    m_file->write((const char*)(&centerOfBox[0]), sizeof(FReal) * typeFReal[2]);
                }
            }
        }
    };
}   // namespace scalfmm::io
#endif   // SCALFMM_TOOLS_FMA_LOADER_HPP
