// --------------------------------
// See LICENCE file at project root
// File : utils/low_rank.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_LOW_RANK_HPP
#define SCALFMM_UTILS_LOW_RANK_HPP

#include <xtensor/xtensor.hpp>

namespace scalfmm::low_rank
{
    template<typename ValueType, typename MatrixKernel, typename TensorViewX, typename TensorViewY>
    inline auto paca(MatrixKernel const& mk, TensorViewX&& X, TensorViewY&& Y, xt::xarray<ValueType> const& weights,
              ValueType epsilon, std::size_t kn, std::size_t km) -> std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>
    {
      const std::size_t nnodes{X.size()};
        std::vector<bool> row_bools(nnodes, true);
        std::vector<bool> col_bools(nnodes, true);
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> U;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> V;

        // initialize rank r
	std::size_t r{0};
        const auto max_r{nnodes};

        // resize U V
        // TODO kn km
        U.resize({nnodes, max_r});
        V.resize({nnodes, max_r});

        // initialize norm
        ValueType norm2s{0.};
        ValueType norm2uv{0.};
        // start partially pivoted aca
        auto evaluate = [&weights, &mk, &X, &Y, kn, km](auto& view, auto Ib, auto Ie, auto Jb, auto Je)
        {
            std::size_t idx{0};
            for(std::size_t j = Jb; j < Je; ++j)
            {
                for(std::size_t i = Ib; i < Ie; ++i)
                {
                    view.at(idx) = weights[i] * weights[j] *
                                   mk.evaluate(X[i], Y[j]).at(kn * MatrixKernel::km + km);
                    ++idx;
                }
            }
        };
        const auto epsilon2 = epsilon*epsilon;
	std::size_t I{0};
	std::size_t J{0};
        do {
            // compute row I and its residual
            auto V_ = xt::view(V, xt::all(), r);
            evaluate(V_, I, I+1, 0, nnodes);
            row_bools[I] = false;
            for(std::size_t l=0; l<r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                V_ -= col_u.at(I)*col_v;
            }
            // find max of residual and argmax
            ValueType val_max{0.};
            for(std::size_t j=0; j<nnodes; ++j)
            {
                const auto val_abs = std::abs(V_.at(j));
                if(col_bools[j] && val_max < val_abs)
                {
                    val_max = val_abs;
                    J = j;
                }
            }
            // find pivot and scale column of V
            const ValueType pivot = ValueType(1.) / V_.at(J);
            V_ *= pivot;

            // compute col J and its residual
            auto U_ = xt::view(U, xt::all(), r);
            evaluate(U_, 0, nnodes, J, J+1);
            col_bools[J] = false;
            for(std::size_t l = 0; l < r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                U_ -= col_v.at(J)*col_u;
            }
            // find max of residual and argmax
            val_max = 0.;
            for(std::size_t i=0; i<nnodes; ++i)
            {
                const auto val_abs = std::abs(U_.at(i));
                if(row_bools[i] && val_max < val_abs)
                {
                    val_max = val_abs;
                    I = i;
                }
            }
            // increment Frobenius norm: |Sk|^2 += |uk|^2 |vk|^2 + 2 sumj ukuj vjvk
            ValueType normuuvv{0.};
            for(std::size_t l=0; l<r; ++l)
            {
                auto col_u = xt::view(U, xt::all(), l);
                auto col_v = xt::view(V, xt::all(), l);
                normuuvv += xt::linalg::vdot(U_, col_u)*xt::linalg::vdot(col_v, V_);
            }
            norm2uv = xt::linalg::vdot(U_, U_) * xt::linalg::vdot(V_, V_);
            norm2s += norm2uv + ValueType(2.)*normuuvv;
            // increment low-rank
            ++r;
        }
        while(norm2uv > epsilon2 * norm2s && r < max_r);

        auto UU = xt::eval(xt::view(U, xt::all(), xt::range(0,r)));
        auto VV = xt::eval(xt::view(V, xt::all(), xt::range(0,r)));
        return std::make_tuple(UU, VV);
    }


    template<typename ValueType, typename MatrixKernel, typename TensorViewX, typename TensorViewY>
    inline auto generate(MatrixKernel const& mk, TensorViewX&& X, TensorViewY&& Y, xt::xarray<ValueType> const& weights,
              ValueType epsilon, std::size_t kn, std::size_t km) -> std::tuple<xt::xtensor<ValueType, 2>, xt::xtensor<ValueType, 2>>
    {
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> U;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> V;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> QU;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> QV;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> RU;
        xt::xtensor<ValueType, 2, xt::layout_type::column_major> RV;

        std::tie(U, V) = paca(mk, std::forward<TensorViewX>(X), std::forward<TensorViewY>(Y), weights, epsilon, kn, km);

        std::tie(QU,RU) = xt::linalg::qr(U);
        std::tie(QV,RV) = xt::linalg::qr(V);

        auto nnodes = U.shape()[0];
        auto rank = U.shape()[1];

        xt::xtensor<ValueType, 2, xt::layout_type::column_major> phi({rank, rank});
        xt::blas::gemm(RU, RV, phi, false, true);

        auto res_svd = xt::linalg::svd(phi);

        auto U_ = std::get<0>(res_svd);
        auto S = std::get<1>(res_svd);
        auto V_ = std::get<2>(res_svd);

        xt::blas::gemm(QV, V_, V, false, true);

        for(std::size_t j = 0; j < rank; ++j)
        {
            for(std::size_t i = 0; i < rank; ++i)
            {
                U_.at(i,j) *= S.at(j);
            }
        }

        xt::blas::gemm(QU, U_, U, false, false);

        //xt::xarray<ValueType> K(std::vector(2, nnodes));
        //for(std::size_t i{0}; i < nnodes; ++i)
        //{
        //    for(std::size_t j{0}; j < nnodes; ++j)
        //    {
        //        K.at(i, j) = weights[i] * weights[j] * mk.evaluate(X[i], Y[j]).at(kn * MatrixKernel::km + km);
        //    }
        //}

        //xt::xarray<ValueType> prod(std::vector(2, nnodes));
        //xt::blas::gemm(U, V, prod, false, true);
        //auto error = K-prod;
        //std::cout << xt::linalg::norm(error) << '\n';

        // unweightening
        for(std::size_t j = 0; j < rank; ++j)
        {
            for(std::size_t i = 0; i < nnodes; ++i)
            {
                U.at(i,j) /= weights[i];
                V.at(i,j) /= weights[i];
            }
        }

        xt::xtensor<ValueType, 2> U_row(U);
        xt::xtensor<ValueType, 2> V_row(V);

        return std::make_tuple(U_row, V_row);
    }

}

#endif  //SCALFMM_UTILS_LOW_RANK_HPP
