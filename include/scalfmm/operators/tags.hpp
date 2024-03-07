// --------------------------------
// See LICENCE file at project root
// File : operators/tags.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_TAGS_HPP
#define SCALFMM_OPERATORS_TAGS_HPP


namespace scalfmm::operators::impl
{
    // Tag dispatching for op
    struct tag
    {
    };
    struct tag_m2m : tag
    {
    };
    struct tag_l2l : tag
    {
    };
    struct tag_m2p : tag
    {
    };
    struct tag_p2m : tag
    {
    };
    struct tag_m2l : tag
    {
    };
    struct tag_p2p : tag
    {
    };
    struct tag_l2p : tag
    {
    };
}   // namespace scalfmm::operators::impl

#endif   // SCALFMM_OPERATORSC_TAGS_HPP
