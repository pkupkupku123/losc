#include "eigen_helper.hpp"
#include <cmath>
#include <losc/curvature.hpp>
#include <losc/exception.hpp>

namespace losc {

void CurvatureV2::C_API_kappa(RefMat kappa2) const
{
    if (!mtx_match_dimension(kappa2, nlo_, nlo_)) {
        throw exception::DimensionError(
            kappa2, nlo_, nlo_,
            "CurvatureV2::kappa(): wrong dimension of the input kappa matrix.");
    }

    // construct absolute overlap under LO.
    LOSCMatrix S_lo(nlo_, nlo_);
    S_lo.setZero();
    // The LO grid value matrix has dimension of (npts, nlo), which could be
    // very large. To limit the memory usage on LO grid value, we build it by
    // blocks. For now we limit the memory usage to be 1GB.
    const size_t block_size =
        1000ULL * 1000ULL * 1000ULL / sizeof(double) / nlo_;
    size_t nBLK = npts_ / block_size;
    const size_t res = npts_ % block_size;
    if (res != 0) {
        nBLK += 1;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    // loop over all the blocks of grid.
    for (size_t n = 0; n < nBLK; ++n) {
        size_t size = block_size;
        if (n == nBLK - 1 && res != 0)
            size = res;

        const auto grid_lo_block =
            grid_lo_.block(n * block_size, 0, size, nlo_);
        const auto wt = grid_weight_.segment(n * block_size, size);

        // sum over current block contribution to the LO overlap.
        for (size_t ip = 0; ip < size; ++ip) {
            const double wt_value = wt[ip];
            for (size_t i = 0; i < nlo_; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    const double pi = grid_lo_block(ip, i);
                    const double pj = grid_lo_block(ip, j);
                    S_lo(i, j) += wt_value * std::abs(pi * pj);
                }
            }
        }
    }
    mtx_to_symmetric(S_lo, "L");

    // build the curvature version 1.
    CurvatureV1 kappa1_man(dfa_info_, 
                           df_pii_, 
                           df_Vpq_inverse_, 
                           grid_lo_,
                           grid_weight_);

    // make sure CurvatureV1.tau be the same as CurvatureV2.tau
    kappa1_man.set_tau(tau_);

    LOSCMatrix kappa1 = kappa1_man.kappa();

    // build LOSC2 kappa matrix:
    // K2[ij] = erf(tau * S[ij]) * sqrt(abs(K1[ii] * K1[jj])) + erfc(tau *
    // S[ij]) * K][ij]
    using std::abs;
    using std::erf;
    using std::erfc;
    using std::sqrt;
    for (size_t i = 0; i < nlo_; ++i) {
        const double K1_ii = kappa1(i, i);
        kappa2(i, i) = K1_ii;
        for (size_t j = 0; j < i; ++j) {
            const double S_ij = S_lo(i, j);
            const double K1_ij = kappa1(i, j);
            const double K1_jj = kappa1(j, j);
            const double f = zeta_ * S_ij;
            kappa2(i, j) = erf(f) * sqrt(abs(K1_ii * K1_jj)) + erfc(f) * K1_ij;
            kappa2(j, i) = kappa2(i, j);
        }
    }
}

/** 
* Added by YeLi 
* @brief Compute the overlap matrix.
*/

void CurvatureV2::C_API_S_lo(RefMat S_lo) const
{
    if (!mtx_match_dimension(S_lo, nlo_, nlo_)) {
        throw exception::DimensionError(
            S_lo, nlo_, nlo_,
            "CurvatureV2::S_lo(): wrong dimension of the input overlap matrix.");
    }

    // construct absolute overlap under LO.
    S_lo.setZero();
    // The LO grid value matrix has dimension of (npts, nlo), which could be
    // very large. To limit the memory usage on LO grid value, we build it by
    // blocks. For now we limit the memory usage to be 1GB.
    const size_t block_size =
        1000ULL * 1000ULL * 1000ULL / sizeof(double) / nlo_;
    size_t nBLK = npts_ / block_size;
    const size_t res = npts_ % block_size;
    if (res != 0) {
        nBLK += 1;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    // loop over all the blocks of grid.
    for (size_t n = 0; n < nBLK; ++n) {
        size_t size = block_size;
        if (n == nBLK - 1 && res != 0)
            size = res;

        const auto grid_lo_block =
            grid_lo_.block(n * block_size, 0, size, nlo_);
        const auto wt = grid_weight_.segment(n * block_size, size);

        // sum over current block contribution to the LO overlap.
        for (size_t ip = 0; ip < size; ++ip) {
            const double wt_value = wt[ip];
            for (size_t i = 0; i < nlo_; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    const double pi = grid_lo_block(ip, i);
                    const double pj = grid_lo_block(ip, j);
                    S_lo(i, j) += wt_value * std::abs(pi * pj);
                }
            }
        }
    }
    mtx_to_symmetric(S_lo, "L");
} 


/**
 * Added by YeLi
 * @brief Construct the kappa2 matrix from kappa1 and S_lo.
*/

void CurvatureV2::kappa1_to_kappa2(ConstRefMat &S_lo, ConstRefMat &kappa1, RefMat kappa2) const
{
    if (!mtx_match_dimension(S_lo, nlo_, nlo_)) {
        throw exception::DimensionError(
            S_lo, nlo_, nlo_,
            "CurvatureV2::kappa1_to_kappa2(): wrong dimension of the input overlap matrix.");
    }

    if (!mtx_match_dimension(kappa1, nlo_, nlo_)) {
        throw exception::DimensionError(
            kappa1, nlo_, nlo_,
            "CurvatureV2::kappa1_to_kappa2(): wrong dimension of the input kappa1 matrix.");
    }

    if (!mtx_match_dimension(kappa2, nlo_, nlo_)) {
        throw exception::DimensionError(
            kappa2, nlo_, nlo_,
            "CurvatureV2::kappa1_to_kappa2(): wrong dimension of the input kappa2 matrix.");
    }
    // build LOSC2 kappa matrix:
    // K2[ij] = erf(tau * S[ij]) * sqrt(abs(K1[ii] * K1[jj])) + erfc(tau *
    // S[ij]) * K][ij]
    using std::abs;
    using std::erf;
    using std::erfc;
    using std::sqrt;
    for (size_t i = 0; i < nlo_; ++i) {
        const double K1_ii = kappa1(i, i);
        kappa2(i, i) = K1_ii;
        for (size_t j = 0; j < i; ++j) {
            const double S_ij = S_lo(i, j);
            const double K1_ij = kappa1(i, j);
            const double K1_jj = kappa1(j, j);
            const double f = zeta_ * S_ij;
            kappa2(i, j) = erf(f) * sqrt(abs(K1_ii * K1_jj)) + erfc(f) * K1_ij;
            kappa2(j, i) = kappa2(i, j);
        }
    }
}



/** Added by YeLi
 * @brief Compute the kappa matrix of J alone.
 */

void CurvatureV2::C_API_kappa_J(RefMat kappa2_J) const
{
    if (!mtx_match_dimension(kappa2_J, nlo_, nlo_)) {
        throw exception::DimensionError(
            kappa2_J, nlo_, nlo_,
            "CurvatureV2::kappa_J(): wrong dimension of the input kappa_J matrix.");
    }

    // construct absolute overlap under LO.
    LOSCMatrix S_lo(nlo_, nlo_);
    C_API_S_lo(S_lo);

    // build the curvature version 1 of J.
    CurvatureV1 kappa1_man(dfa_info_,
                           df_pii_, 
                           df_Vpq_inverse_, 
                           grid_lo_,
                           grid_weight_);
    LOSCMatrix kappa1_J = kappa1_man.kappa_J();
    kappa1_to_kappa2(S_lo, kappa1_J, kappa2_J);
}

/**
 * Added by YeLi
 * @brief Compute the kappa matrix of LDA exchange alone.
 */

void CurvatureV2::C_API_kappa_LDAX(RefMat kappa2_LDAX) const
{
    if (!mtx_match_dimension(kappa2_LDAX, nlo_, nlo_)) {
        throw exception::DimensionError(
            kappa2_LDAX, nlo_, nlo_,
            "CurvatureV2::kappa_LDAX(): wrong dimension of the input kappa_LDAX matrix.");
    }

    // construct absolute overlap under LO.
    LOSCMatrix S_lo(nlo_, nlo_);
    C_API_S_lo(S_lo);

    // build the curvature version 1 of LDA exchange alone.
    CurvatureV1 kappa1_man(dfa_info_, 
                           df_pii_, 
                           df_Vpq_inverse_, 
                           grid_lo_,
                           grid_weight_);
    LOSCMatrix kappa1_LDAX = kappa1_man.kappa_LDAX();
    kappa1_to_kappa2(S_lo, kappa1_LDAX, kappa2_LDAX);
}

}// namespace losc
