#ifndef EXAMPLE_INTERACTION_RESULTS_HPP
#define EXAMPLE_INTERACTION_RESULTS_HPP
#include <array>

// circle-100_target.fma ; circle-100_source.fma
auto st_get_symbolic_list_p2p(std::size_t morton) 
{
    using array_type = std::array<std::size_t, 9>;
    switch(morton)
    {
    case 1:
        return array_type{1, 2, 3, 4};
    case 2:
        return array_type{1, 2, 3, 8};
    case 3:
        return array_type{1, 2, 3, 4, 8};
    case 4:
        return array_type{1, 3, 4, 5};
    case 5:
        return array_type{4, 5, 16};
    case 8:
        return array_type{2, 3, 8, 10};
    case 10:
        return array_type{8, 10, 32};
    case 16:
        return array_type{5, 16, 17};
    case 17:
        return array_type{16, 17, 20, 22};
    case 20:
        return array_type{17, 20, 22, 23};
    case 22:
        return array_type{17, 20, 22, 23, 29};
    case 23:
        return array_type{20, 22, 23, 29};
    case 29:
        return array_type{22, 23, 29, 31};
    case 31:
        return array_type{29, 31, 53};
    case 32:
        return array_type{10, 32, 34};
    case 34:
        return array_type{32, 34, 40, 41};
    case 40:
        return array_type{34, 40, 41, 43};
    case 41:
        return array_type{34, 40, 41, 43, 46};
    case 43:
        return array_type{40, 41, 43, 46};
    case 46:
        return array_type{41, 43, 46, 47};
    case 47:
        return array_type{46, 47, 58};
    case 53:
        return array_type{31, 53, 55};
    case 55:
        return array_type{53, 55, 60, 61};
    case 58:
        return array_type{47, 58, 59};
    case 59:
        return array_type{58, 59, 60, 62};
    case 60:
        return array_type{55, 59, 60, 61, 62};
    case 61:
        return array_type{55, 60, 61, 62};
    case 62:
        return array_type{59, 60, 61, 62};
    default:
        return array_type{};
    }
}
auto st_get_symbolic_list_m2l(std::size_t morton, const int level)
{
    using array_type = std::array<std::size_t, 27>;
    switch(level)
    {
    case 3:
        switch(morton)
        {
        case 1:
            return array_type{5, 8, 10};
        case 2:
            return array_type{4, 5, 10};
        case 3:
            return array_type{5, 10};
        case 4:
            return array_type{2, 8, 10, 16, 17};
        case 5:
            return array_type{1, 2, 3, 8, 10, 17};
        case 8:
            return array_type{1, 4, 5, 32, 34};
        case 10:
            return array_type{1, 2, 3, 4, 5, 34};
        case 16:
            return array_type{4, 20, 22, 23, 29, 31};
        case 17:
            return array_type{4, 5, 23, 29, 31};
        case 20:
            return array_type{16, 29, 31};
        case 22:
            return array_type{16, 31};
        case 23:
            return array_type{16, 17, 31};
        case 29:
            return array_type{16, 17, 20, 53, 55};
        case 31:
            return array_type{16, 17, 20, 22, 23, 55};
        case 32:
            return array_type{8, 40, 41, 43, 46, 47};
        case 34:
            return array_type{8, 10, 43, 46, 47};
        case 40:
            return array_type{32, 46, 47};
        case 41:
            return array_type{32, 47};
        case 43:
            return array_type{32, 34, 47};
        case 46:
            return array_type{32, 34, 40, 58, 59};
        case 47:
            return array_type{32, 34, 40, 41, 43, 59};
        case 53:
            return array_type{29, 58, 59, 60, 61, 62};
        case 55:
            return array_type{29, 31, 58, 59, 62};
        case 58:
            return array_type{46, 53, 55, 60, 61, 62};
        case 59:
            return array_type{46, 47, 53, 55, 61};
        case 60:
            return array_type{53, 58};
        case 61:
            return array_type{53, 58, 59};
        case 62:
            return array_type{53, 55, 58};
        default:
            return array_type{};
        }
    case 2:
        switch(morton)
        {
        case 0:
            return array_type{4, 5, 7, 8, 10, 11, 13, 14, 15};
        case 1:
            return array_type{5, 7, 8, 10, 11, 13, 14, 15};
        case 2:
            return array_type{4, 5, 7, 10, 11, 13, 14, 15};
        case 4:
            return array_type{0, 2, 8, 10, 11, 13, 14, 15};
        case 5:
            return array_type{0, 1, 2, 8, 10, 11, 13, 14, 15};
        case 7:
            return array_type{0, 1, 2, 8, 10, 11, 14, 15};
        case 8:
            return array_type{0, 1, 4, 5, 7, 13, 14, 15};
        case 10:
            return array_type{0, 1, 2, 4, 5, 7, 13, 14, 15};
        case 11:
            return array_type{0, 1, 2, 4, 5, 7, 13, 15};
        case 13:  
            return array_type{0, 1, 2, 4, 5, 8, 10, 11};
        case 14:
            return array_type{0, 1, 2, 4, 5, 7, 8, 10};
        case 15:
            return array_type{0, 1, 2, 4, 5, 7, 8, 10, 11};

        default:
            return array_type{};
        }
    default:
        return array_type{};
    }
}


// circle-100_target.fma 
auto t_get_symbolic_list_p2p(std::size_t morton) 
{
    using array_type = std::array<std::size_t, 9>;
    switch(morton)
    {
    case 1:
        return array_type{2, 3, 4};
    case 2:
        return array_type{1, 3, 8};
    case 3:
        return array_type{1, 2, 4, 8};
    case 4:
        return array_type{1, 3, 5};
    case 5:
        return array_type{4, 16};
    case 8:
        return array_type{2, 3, 10};
    case 10:
        return array_type{8, 32};
    case 16:
        return array_type{5, 17};
    case 17:
        return array_type{16, 20, 22};
    case 20:
        return array_type{17, 22, 23};
    case 22:
        return array_type{17, 20, 23, 29};
    case 23:
        return array_type{20, 22, 29};
    case 29:
        return array_type{22, 23, 31};
    case 31:
        return array_type{29, 53};
    case 32:
        return array_type{10, 34};
    case 34:
        return array_type{32, 40, 41};
    case 40:
        return array_type{34, 41, 43};
    case 41:
        return array_type{34, 40, 43, 46};
    case 43:
        return array_type{40, 41, 46};
    case 46:
        return array_type{41, 43, 47};
    case 47:
        return array_type{46, 58};
     case 53:
         return array_type{31, 55};
    case 55:
        return array_type{53, 60, 61};
    case 58:
        return array_type{47, 59};
    case 59:
        return array_type{58, 60, 62};
    case 60:
        return array_type{55, 59, 61, 62};
    case 61:
        return array_type{55, 60, 62};
    case 62:
        return array_type{59, 60, 61};
    default:
        return array_type{};
    }
}
auto t_get_symbolic_list_m2l(std::size_t morton, const int level)
{
    using array_type = std::array<std::size_t, 27>;
    switch(level)
    {
    case 3:
        switch(morton)
        {
        case 1:
            return array_type{5, 8, 10};
        case 2:
            return array_type{4, 5, 10};
        case 3:
            return array_type{5, 10};
        case 4:
            return array_type{2, 8, 10, 16, 17};
        case 5:
            return array_type{1, 2, 3, 8, 10, 17};
        case 8:
            return array_type{1, 4, 5, 32, 34};
        case 10:
            return array_type{1, 2, 3, 4, 5, 34};
        case 16:
            return array_type{4, 20, 22, 23, 29, 31};
        case 17:
            return array_type{4, 5, 23, 29, 31};
        case 20:
            return array_type{16, 29, 31};
        case 22:
            return array_type{16, 31};
        case 23:
            return array_type{16, 17, 31};
        case 29:
            return array_type{16, 17, 20, 53, 55};
        case 31:
            return array_type{16, 17, 20, 22, 23, 55};
        case 32:
            return array_type{8, 40, 41, 43, 46, 47};
        case 34:
            return array_type{8, 10, 43, 46, 47};
        case 40:
            return array_type{32, 46, 47};
        case 41:
            return array_type{32, 47};
        case 43:
            return array_type{32, 34, 47};
        case 46:
            return array_type{32, 34, 40, 58, 59};
        case 47:
            return array_type{32, 34, 40, 41, 43, 59};
        case 53:
            return array_type{29, 58, 59, 60, 61, 62};
        case 55:
            return array_type{29, 31, 58, 59, 62};
        case 58:
            return array_type{46, 53, 55, 60, 61, 62};
        case 59:
            return array_type{46, 47, 53, 55, 61};
        case 60:
            return array_type{53, 58};
        case 61:
            return array_type{53, 58, 59};
        case 62:
            return array_type{53, 55, 58};
        default:
            return array_type{};
        }
    case 2:
        switch(morton)
        {
        case 0:
            return array_type{4, 5, 7, 8, 10, 11, 13, 14, 15};
        case 1:
            return array_type{5, 7, 8, 10, 11, 13, 14, 15};
        case 2:
            return array_type{4, 5, 7, 10, 11, 13, 14, 15};
        case 4:
            return array_type{0, 2, 8, 10, 11, 13, 14, 15};
        case 5:
            return array_type{0, 1, 2, 8, 10, 11, 13, 14, 15};
        case 7:
            return array_type{0, 1, 2, 8, 10, 11, 14, 15};
        case 8:
            return array_type{0, 1, 4, 5, 7, 13, 14, 15};
        case 10:
            return array_type{0, 1, 2, 4, 5, 7, 13, 14, 15};
        case 11:
            return array_type{0, 1, 2, 4, 5, 7, 13, 15};
        case 13:
            return array_type{0, 1, 2, 4, 5, 8, 10, 11};
        case 14:
            return array_type{0, 1, 2, 4, 5, 7, 8, 10};
        case 15:
            return array_type{0, 1, 2, 4, 5, 7, 8, 10, 11};

        default:
            return array_type{};
        }
    default:
        return array_type{};
    }
}
#endif
