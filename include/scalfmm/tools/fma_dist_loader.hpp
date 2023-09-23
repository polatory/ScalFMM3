// --------------------------------
// See LICENCE file at project root
// File : tools/fma_loader.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_FMA_MPI_LOADER_HPP
#define SCALFMM_TOOLS_FMA_MPI_LOADER_HPP
#include <cpp_tools/parallel_manager/parallel_manager.hpp>
#include <cstdlib>
#include <vector>
#include <array>
#include <iostream>
#include <limits>
#include <string>

#include "scalfmm/tools/fma_loader.hpp"

namespace scalfmm::io
{
    ///
    /// \brief The FMpiFmaGenericLoader class
    ///
    template<class FReal, int Dimension = 3>
    class DistFmaGenericLoader : public FFmaGenericLoader<FReal, Dimension>
    {
      protected:
        using FFmaGenericLoader<FReal, Dimension>::m_nbParticles;
        using FFmaGenericLoader<FReal, Dimension>::m_file;
        using FFmaGenericLoader<FReal, Dimension>::m_typeData;
        using FFmaGenericLoader<FReal, Dimension>::m_verbose;
        using MPI_Offset = std::size_t;

        std::size_t m_local_number_of_particles;   ///< Number of particles that the calling process will manage
        MPI_Offset m_idxParticles;                 //
        std::size_t m_start;                       ///< number of my first parts in file
        size_t m_headerSize;
        const cpp_tools::parallel_manager::parallel_manager* m_parallelManager;

      public:
        DistFmaGenericLoader(const std::string inFilename, const cpp_tools::parallel_manager::parallel_manager& para, const bool verbose = false)
          : FFmaGenericLoader<FReal, Dimension>(inFilename, verbose)
          , m_local_number_of_particles(0)
          , m_idxParticles(0)
          , m_headerSize(0)
          , m_parallelManager(&para)
        {
            // the header is already read by the constructor of FFmaGenericLoader
            std::size_t bloc = m_nbParticles / FReal(m_parallelManager->get_num_processes());
            std::cout << "bloc: " << bloc << " part " << m_nbParticles << " nP  "
                      << FReal(m_parallelManager->get_num_processes()) << std::endl;
            // Determine the number of particles to read
            auto rank = m_parallelManager->get_process_id();
            std::size_t startPart = bloc * rank ;
            std::size_t endPart = (rank+1  == m_parallelManager->get_num_processes()) ? m_nbParticles : startPart + bloc;

            this->m_start = startPart;
            this->m_local_number_of_particles = endPart - startPart;
            //std::cout << " startPart " << startPart << " endPart " << endPart << std::endl;
            //            std::cout << "Proc " << m_parallelManager->get_process_id() << " will hold " <<
            //            m_local_number_of_particles
            //                      << std::endl;
            // Skip the header
            if(this->m_binaryFile)
            {
                // This is header size in bytes
                //   MEANING :      sizeof(FReal)+nbAttr, nb of parts, boxWidth+boxCenter
                m_headerSize = FFmaGenericLoader<FReal, Dimension>::get_header_size();

                // To this header size, we had the parts that belongs to proc on my left
                m_file->seekg(m_headerSize + startPart * m_typeData[1] * sizeof(FReal));
            }
            else
            {
                // First finish to read the current line
                m_file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                for(int i = 0; i < startPart; ++i)
                {
                    m_file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
        }

        ~DistFmaGenericLoader() {}

        auto getMyNumberOfParticles() const { return m_local_number_of_particles; }

        auto getStart() const { return m_start; }

        /**
         * Given an index, get the one particle from this index
         */
        void fill1Particle(FReal* data, std::size_t indexInFile)
        {
            m_file->seekg(m_headerSize +
                          (int(indexInFile) * FFmaGenericLoader<FReal>::getNbRecordPerline() * sizeof(FReal)));
            m_file->read((char*)data, FFmaGenericLoader<FReal>::getNbRecordPerline() * sizeof(FReal));
        }
    };
    /**
     *
     * \brief Writes a set of distributed particles to an FMA formated file.
     *
     * The file may be in ASCII or binary mode. The example below shows how to use the class.
     *
     * \code
     * // Instanciate the writer with a binary fma file (extension .bfma).
     * \endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * file
     */
    template<class FReal>
    class DistFmaGenericWriter : public FFmaGenericWriter<FReal>
    {
      protected:
        const cpp_tools::parallel_manager::parallel_manager* m_parallelManager;
        bool _writeDone;
        int m_headerSize;
        int _nbDataTowritePerRecord;      //< number of data to write for one particle
        std::size_t _numberOfParticles;   //< number of particle (global) to write in the file
        //
        using FFmaGenericWriter<FReal>::m_file;
#ifdef SCALFMM_USE_MPI
        MPI_File _mpiFile;   //< MPI pointer on data file (write mode)
#endif
      public:
        /**
         * This constructor opens a file to be written to.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         *
         * @param filename the name of the file to open.
         */
        DistFmaGenericWriter(const std::string inFilename, const cpp_tools::parallel_manager::parallel_manager& para)
          : FFmaGenericWriter<FReal>(inFilename)
          , m_parallelManager(&para)
          , _writeDone(false)
          , m_headerSize(0)
          , _nbDataTowritePerRecord(8)
          , _numberOfParticles(0)
        {
#ifdef SCALFMM_USE_MPI
            if(!this->isBinary())
            {
                std::cout << "DistFmaGenericWriter only works with binary file (.bfma)." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            auto comm = m_parallelManager->get_communicator();

            int fileIsOpen = MPI_File_open(comm.get_comm(), const_cast<char*>(inFilename.c_str()),
                                           MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &_mpiFile);
            // Is it open?
            if(fileIsOpen != MPI_SUCCESS)
            {
                std::cerr << "Cannot create parallel file, DistFmaGenericWriter constructeur abort." << std::endl;
                std::exit(EXIT_FAILURE);
                return;
            }
#endif
        }
        /**
         * Writes the header of FMA file.
         *
         * Should be used if we write the particles with writeArrayOfReal method
         *
         * @param centerOfBox      The center of the Box (POINT_T class)
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
            //      * \code
            //      *   DatatypeSize  Number_of_record_per_line
            //      *   NB_particles  half_Box_width  Center_X  Center_Y  Center_Z
            //      *   Particle_values
            //      * \endcode
            std::array<unsigned int, 4> typeFReal = {dataType, nbDataPerRecord, dimension, nb_input_values};
            FReal x = boxWidth * FReal(0.5);
            m_headerSize = 0;
            _nbDataTowritePerRecord = nbDataPerRecord;
            _numberOfParticles = nbParticles;
            if(m_parallelManager->master())
            {
                FFmaGenericWriter<FReal>::writerBinaryHeader(centerOfBox, boxWidth, nbParticles, typeFReal.data(), 4);
                std::cout << "centerOfBox " << centerOfBox << " boxWidth " << boxWidth << " nbParticles " << nbParticles
                          << " dataType " << dataType << " nbDataPerRecord " << nbDataPerRecord << " dimension "
                          << dimension << " nb_input_values " << nb_input_values << std::endl;
#ifdef SCALFMM_USE_MPI
                for(auto a: typeFReal)
                {
                    std::cout << "typeFReal " << a << std::endl;
                }
                int sizeType = 0;
                int ierr = 0;
                auto mpiInt64 = cpp_tools::parallel_manager::mpi::get_datatype<std::size_t>();
                auto mpiUInt = cpp_tools::parallel_manager::mpi::get_datatype<unsigned int>();
                auto mpiReal = cpp_tools::parallel_manager::mpi::get_datatype<FReal>();
                //
                if(typeFReal[0] != sizeof(FReal))
                {
                    std::cout << " ssss Size of elements in part file " << typeFReal[0]
                              << " is different from size of FReal " << sizeof(FReal) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                //  ierr = MPI_File_write_at(_mpiFile, 0, typeFReal.data(), typeFReal.size(), mpiUInt,
                //  MPI_STATUS_IGNORE);
                MPI_Type_size(mpiUInt, &sizeType);
                m_headerSize += sizeType * typeFReal.size();
                //     ierr = MPI_File_write_at(_mpiFile, m_headerSize, &nbParticles, 1, mpiInt64, MPI_STATUS_IGNORE);
                MPI_Type_size(mpiInt64, &sizeType);
                m_headerSize += sizeType * 1;

                //                std::array<FReal, 1 + POINT_T::dimension> boxSim{};
                //                boxSim[0] = boxWidth;
                //                for(int d = 0; d < POINT_T::dimension; ++d)
                //                {
                //                    boxSim[d + 1] = centerOfBox[d];
                //                }
                //     ierr = MPI_File_write_at(_mpiFile, m_headerSize, &boxSim[0], 4, mpiReal, MPI_STATUS_IGNORE);
                MPI_Type_size(mpiReal, &sizeType);
                m_headerSize += sizeType * (1 + POINT_T::dimension);
                // Build the header offset
                std::cout << " headerSize " << m_headerSize << std::endl;
                FFmaGenericWriter<FReal>::close();
#endif
            }
#ifdef SCALFMM_USE_MPI
            auto comm = m_parallelManager->get_communicator();

            comm.bcast(&m_headerSize, 1, MPI_INT, 0);
            //  MPI_Bcast(&_headerSize, 1, MPI_INT, 0, m_parallelManager->global().getComm());
            std::cout << "  _headerSize  " << m_headerSize << std::endl;
#endif

            //  MPI_File_close(&_mpiFile);
        }
        ~DistFmaGenericWriter()
        { /* MPI_File_close(&_mpiFile);*/
        }

        /**
         *  Write all for all particles the position, physical values, potential and forces
         *
         * @param myOctree the octree
         * @param nbParticlesnumber of particles
         * @param mortonLeafDistribution the morton distribution of the leaves (this is a vecor of size 2* the
         number of
         * MPI processes
         *
         */
        template<typename OCTREECLASS>
        void writeFromTree(const OCTREECLASS& myOctree, const std::size_t& nbParticles)
        {
            //            //
            //            // Write the header
            //            int sizeType = 0, ierr = 0;
            //            FReal tt = 0.0;
            //            MPI_Datatype mpistd::size_t_t = m_parallelManager->GetType(nbParticles);
            //            MPI_Datatype mpiFReal_t = m_parallelManager->GetType(tt);
            //            MPI_Type_size(mpiFReal_t, &sizeType);
            //            int myRank = m_parallelManager->global().processId();
            //            _headerSize = 0;
            //            //
            //            unsigned int typeFReal[2] = {sizeof(FReal), static_cast<unsigned
            //            int>(_nbDataTowritePerRecord)}; if(myRank == 0)
            //            {
            //                ierr = MPI_File_write_at(_mpiFile, 0, &typeFReal, 2, MPI_INT, MPI_STATUS_IGNORE);
            //            }
            //            MPI_Type_size(MPI_INT, &sizeType);
            //            _headerSize += sizeType * 2;
            //            if(myRank == 0)
            //            {
            //                ierr = MPI_File_write_at(_mpiFile, _headerSize, &nbParticles, 1, mpistd::size_t_t,
            //                MPI_STATUS_IGNORE);
            //            }
            //            MPI_Type_size(mpistd::size_t_t, &sizeType);
            //            _headerSize += sizeType * 1;
            //            auto centerOfBox = myOctree.getBoxCenter();
            //            FReal boxSim[4] = {myOctree.getBoxWidth() * 0.5, centerOfBox.getX(), centerOfBox.getX(),
            //                               centerOfBox.getX()};

            //            if(myRank == 0)
            //            {
            //                ierr = MPI_File_write_at(_mpiFile, _headerSize, &boxSim[0], 4, mpiFReal_t,
            //                MPI_STATUS_IGNORE);
            //            }
            //            if(ierr > 0)
            //            {
            //                std::cerr << "Error during the construction of the header in "
            //                             "FMpiFmaGenericWriter::writeDistributionOfParticlesFromOctree"
            //                          << std::endl;
            //            }
            //            MPI_Type_size(mpiFReal_t, &sizeType);
            //            _headerSize += sizeType * 4;
            //            //
            //            // Construct the local number of particles on my process
            //            std::size_t nbLocalParticles = 0, maxPartLeaf = 0;
            //            MortonIndex starIndex = mortonLeafDistribution[2 * myRank],
            //                        endIndex = mortonLeafDistribution[2 * myRank + 1];
            //            myOctree.template forEachCellLeaf<typename OCTREECLASS::LeafClass_T>(
            //              [&](typename OCTREECLASS::GroupSymbolCellClass_T* gsymb,
            //                  typename OCTREECLASS::GroupCellUpClass_T* /* gmul */,
            //                  typename OCTREECLASS::GroupCellDownClass_T* /* gloc */,
            //                  typename OCTREECLASS::LeafClass_T* leafTarget) {
            //                  if(!(gsymb->getMortonIndex() < starIndex || gsymb->getMortonIndex() > endIndex))
            //                  {
            //                      auto n = leafTarget->getNbParticles();
            //                      nbLocalParticles += n;
            //                      maxPartLeaf = std::max(maxPartLeaf, n);
            //                  }
            //              });
            //            std::vector<FReal> particles(maxPartLeaf * _nbDataTowritePerRecord);
            //            // Build the offset for eaxh processes
            //            std::size_t before = 0;   // Number of particles before me (rank < myrank)
            //            MPI_Scan(&nbLocalParticles, &before, 1, mpistd::size_t_t, MPI_SUM,
            //            m_parallelManager->global().getComm()); before -= nbLocalParticles; MPI_Offset offset =
            //            _headerSize + sizeType * _nbDataTowritePerRecord * before;
            //            //
            //            // Write particles in file
            //            myOctree.template forEachCellLeaf<typename OCTREECLASS::LeafClass_T>(
            //              [&](typename OCTREECLASS::GroupSymbolCellClass_T* gsymb,
            //                  typename OCTREECLASS::GroupCellUpClass_T* /* gmul */,
            //                  typename OCTREECLASS::GroupCellDownClass_T* /* gloc */,
            //                  typename OCTREECLASS::LeafClass_T* leafTarget) {
            //                  if(!(gsymb->getMortonIndex() < starIndex || gsymb->getMortonIndex() > endIndex))
            //                  {
            //                      const std::size_t nbPartsInLeaf = leafTarget->getNbParticles();
            //                      const FReal* const posX = leafTarget->getPositions()[0];
            //                      const FReal* const posY = leafTarget->getPositions()[1];
            //                      const FReal* const posZ = leafTarget->getPositions()[2];
            //                      const FReal* const physicalValues = leafTarget->getPhysicalValues();
            //                      const FReal* const forceX = leafTarget->getForcesX();
            //                      const FReal* const forceY = leafTarget->getForcesY();
            //                      const FReal* const forceZ = leafTarget->getForcesZ();
            //                      const FReal* const potential = leafTarget->getPotentials();
            //                      for(int i = 0, k = 0; i < nbPartsInLeaf; ++i, k += _nbDataTowritePerRecord)
            //                      {
            //                          particles[k] = posX[i];
            //                          particles[k + 1] = posY[i];
            //                          particles[k + 2] = posZ[i];
            //                          particles[k + 3] = physicalValues[i];
            //                          particles[k + 4] = potential[i];
            //                          particles[k + 5] = forceX[i];
            //                          particles[k + 6] = forceY[i];
            //                          particles[k + 7] = forceZ[i];
            //                      }
            //                      MPI_File_write_at(_mpiFile, offset, particles.data(),
            //                                        static_cast<int>(_nbDataTowritePerRecord * nbPartsInLeaf),
            //                                        mpiFReal_t, MPI_STATUS_IGNORE);
            //                      offset += sizeType * _nbDataTowritePerRecord * nbPartsInLeaf;
            //                  }
            //              });

#ifdef SCALFMM_USE_MPI
            MPI_File_close(&_mpiFile);
#endif
        }

        //        /**
        //         *  Write all for all particles the position, physical values, potential and forces
        //         *
        //         * @param myOctree the octree
        //         * @param nbParticlesnumber of particles
        //         * @param nbLocalParticles number of local particles (on the MPI processus
        //         * @param mortonLeafDistribution the morton distribution of the leaves (this is a vector of size 2*
        //         the number
        //         * of MPI processes
        //         *
        //         */
        //        template<class OCTREECLASS>
        //        void writeDistributionOfParticlesFromOctree(OCTREECLASS& myOctree, const std::size_t& nbParticles,
        //                                                    const std::size_t& nbLocalParticles,
        //                                                    const std::vector<MortonIndex>& mortonLeafDistribution)
        //        {
        //            //
        //            // Write the header
        //            int sizeType = 0, ierr = 0;
        //            FReal tt{};

        //            MPI_Datatype mpistd::size_t_t = m_parallelManager->GetType(nbParticles);
        //            MPI_Datatype mpiFReal_t = m_parallelManager->GetType(tt);
        //            MPI_Type_size(mpiFReal_t, &sizeType);
        //            int myRank = m_parallelManager->global().processId();
        //            _headerSize = 0;
        //            //
        //            unsigned int typeFReal[2] = {sizeof(FReal), static_cast<unsigned int>(_nbDataTowritePerRecord)};
        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, 0, &typeFReal, 2, MPI_INT, MPI_STATUS_IGNORE);
        //            }
        //            MPI_Type_size(MPI_INT, &sizeType);
        //            _headerSize += sizeType * 2;
        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, _headerSize, const_cast<std::size_t*>(&nbParticles), 1,
        //                mpistd::size_t_t,
        //                                         MPI_STATUS_IGNORE);
        //            }
        //            MPI_Type_size(mpistd::size_t_t, &sizeType);
        //            _headerSize += sizeType * 1;
        //            auto centerOfBox = myOctree.getBoxCenter();
        //            FReal boxSim[4] = {myOctree.getBoxWidth() * 0.5, centerOfBox.getX(), centerOfBox.getX(),
        //                               centerOfBox.getX()};

        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, _headerSize, &boxSim[0], 4, mpiFReal_t, MPI_STATUS_IGNORE);
        //            }
        //            if(ierr > 0)
        //            {
        //                std::cerr << "Error during the construction of the header in "
        //                             "FMpiFmaGenericWriter::writeDistributionOfParticlesFromOctree"
        //                          << std::endl;
        //            }
        //            MPI_Type_size(mpiFReal_t, &sizeType);
        //            _headerSize += sizeType * 4;
        //            //
        //            // Construct the local number of particles on my process
        //            std::size_t maxPartLeaf = 0, nn = 0;
        //            MortonIndex starIndex = mortonLeafDistribution[2 * myRank],
        //                        endIndex = mortonLeafDistribution[2 * myRank + 1];
        //            //  myOctree.template forEachCellLeaf<typename OCTREECLASS::LeafClass_T >(
        //            myOctree.forEachCellLeaf(
        //              [&](typename OCTREECLASS::CellClassType* cell, typename OCTREECLASS::LeafClass_T* leaf) {
        //                  if(!(cell->getMortonIndex() < starIndex || cell->getMortonIndex() > endIndex))
        //                  {
        //                      auto n = leaf->getSrc()->getNbParticles();
        //                      maxPartLeaf = std::max(maxPartLeaf, n);
        //                      nn += n;
        //                  }
        //              });
        //            std::cout << "  nn " << nn << "  should be " << nbLocalParticles << std::endl;
        //            std::vector<FReal> particles(maxPartLeaf * _nbDataTowritePerRecord);
        //            // Build the offset for eaxh processes
        //            std::size_t before = 0;   // Number of particles before me (rank < myrank)
        //            MPI_Scan(const_cast<std::size_t*>(&nbLocalParticles), &before, 1, mpistd::size_t_t, MPI_SUM,
        //                     m_parallelManager->global().getComm());
        //            before -= nbLocalParticles;
        //            MPI_Offset offset = _headerSize + sizeType * _nbDataTowritePerRecord * before;
        //            //
        //            // Write particles in file
        //            myOctree.forEachCellLeaf(
        //              [&](typename OCTREECLASS::CellClassType* cell, typename OCTREECLASS::LeafClass_T* leaf) {
        //                  if(!(cell->getMortonIndex() < starIndex || cell->getMortonIndex() > endIndex))
        //                  {
        //                      const std::size_t nbPartsInLeaf = leaf->getTargets()->getNbParticles();
        //                      const FReal* const posX = leaf->getTargets()->getPositions()[0];
        //                      const FReal* const posY = leaf->getTargets()->getPositions()[1];
        //                      const FReal* const posZ = leaf->getTargets()->getPositions()[2];
        //                      const FReal* const physicalValues = leaf->getTargets()->getPhysicalValues();
        //                      const FReal* const forceX = leaf->getTargets()->getForcesX();
        //                      const FReal* const forceY = leaf->getTargets()->getForcesY();
        //                      const FReal* const forceZ = leaf->getTargets()->getForcesZ();
        //                      const FReal* const potential = leaf->getTargets()->getPotentials();
        //                      for(int i = 0, k = 0; i < nbPartsInLeaf; ++i, k += _nbDataTowritePerRecord)
        //                      {
        //                          particles[k] = posX[i];
        //                          particles[k + 1] = posY[i];
        //                          particles[k + 2] = posZ[i];
        //                          particles[k + 3] = physicalValues[i];
        //                          particles[k + 4] = potential[i];
        //                          particles[k + 5] = forceX[i];
        //                          particles[k + 6] = forceY[i];
        //                          particles[k + 7] = forceZ[i];
        //                      }
        //                      MPI_File_write_at(_mpiFile, offset, particles.data(),
        //                                        static_cast<int>(_nbDataTowritePerRecord * nbPartsInLeaf), mpiFReal_t,
        //                                        MPI_STATUS_IGNORE);
        //                      offset += sizeType * _nbDataTowritePerRecord * nbPartsInLeaf;
        //                  }
        //              });
        //#ifdef TODO
        //#endif
        //            MPI_File_close(&_mpiFile);
        //        }
    };
}   // namespace scalfmm::tools
#endif   // FMPIFMAGENERICLOADER_HPP
