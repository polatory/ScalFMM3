// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/priorities.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP
#define SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP

#ifdef __clang__
#define ACCESS_DEP(V,i) V[i]
#elif defined(__GNUC__)
#define ACCESS_DEP(V,i) V.at(i)
#else
#warning "MACRO FOR DEPENDENCIES : OTHER COMPILER TO CHECK (NOT GCC & CLANG) !"
#define ACCESS_DEP(V,i) V.at(i)
#endif

namespace scalfmm::algorithms::omp
{
    struct priorities
    {
        static constexpr int max = 10;
        static constexpr int p2m = 9;
        static constexpr int m2m = 8;
        //static constexpr int m2l_high = 7,
        static constexpr int m2l = 7;
        static constexpr int l2l = 6;
        static constexpr int p2p_big = 5;
        static constexpr int l2p = 3;
        static constexpr int p2p_small = 2;
    };
    
    //struct priorities
    //{
    //    constexpr priorities(std::size_t tree_height)
    //      : _th(tree_height)
    //    {
    //      if(tree_height > 2)
    //      {
    //        int inc_prio{0};
    //        _p2m_send = inc_prio++;
    //        _p2m = inc_prio++;
    //        _m2m_send = inc_prio++;
    //        _m2m = inc_prio++;
    //        _p2p = inc_prio++;
    //        _m2l = inc_prio++;
    //        _l2l = inc_prio++;
    //        inc_prio += (tree_height-3)-1;
    //        inc_prio += (tree_height-3)-1;
    //        inc_prio += (tree_height-3)-1;
    //        _p2p_out = inc_prio++;
    //        _m2l_last_level = inc_prio++;
    //        _l2p = inc_prio++;
    //        _max_prio = inc_prio;
    //      }
    //    }

    //    int p2m()            const noexcept { return _p2m; }
    //    int p2m_send()       const noexcept { return _p2m_send; }
    //    int m2m_send()       const noexcept { return _m2m_send; }
    //    int m2l()            const noexcept { return _m2l; }
    //    int m2l_last_level() const noexcept { return _m2l_last_level; }
    //    int l2l()            const noexcept { return _l2l; }
    //    int p2p()            const noexcept { return _p2p; }
    //    int p2p_out()        const noexcept { return _p2p_out; }
    //    int l2p()            const noexcept { return _l2p; }
    //    int max_prio()       const noexcept { return _max_prio; }

    //    std::size_t _th{0};
    //    int _p2m{0};
    //    int _p2m_send{0};
    //    int _m2m_send{0};
    //    int _m2m{0};
    //    int _m2l{0};
    //    int _m2l_last_level{0};
    //    int _l2l{0};
    //    int _p2p{0};
    //    int _p2p_out{0};
    //    int _l2p{0};
    //    int _max_prio{0};
    //};

    inline int scale_prio(int prio, std::size_t level)
    {
      return prio+(level-2)*3;
    }
}

#endif // SCALFMM_ALGORITHMS_OMP_PRIORITIES_HPP
