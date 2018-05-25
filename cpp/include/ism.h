//
// Created by xiangyu on 5/16/18.
//

#ifndef CPP_INCLUDE_ISM_H
#define CPP_INCLUDE_ISM_H

#include "include/acl.h"

namespace Nice {
template<class T>
class ISM : public ACL<T> {
 public:
  using ACL<T>::c_;
  using ACL<T>::q_;
  using ACL<T>::n_;
  using ACL<T>::d_;
  using ACL<T>::lambda_;
  using ACL<T>::alpha_;
  using ACL<T>::kernel_type_;
  using ACL<T>::constant_;
  using ACL<T>::u_converge_;
  using ACL<T>::w_converge_;
  using ACL<T>::u_w_converge_;
  using ACL<T>::threshold1_;
  using ACL<T>::threshold2_;
  using ACL<T>::x_matrix_;
  using ACL<T>::w_matrix_;
  using ACL<T>::pre_w_matrix_;
  using ACL<T>::u_matrix_;
  using ACL<T>::pre_u_matrix_;
  using ACL<T>::verbose_;
  using ACL<T>::debug_;
  using ACL<T>::max_time_exceeded_;
  using ACL<T>::max_time_;
  using ACL<T>::method_;
  using ACL<T>::mode_;
  using ACL<T>::clustering_result_;
  using ACL<T>::u_matrix_normalized_;
  using ACL<T>::y_matrix_;
  using ACL<T>::y_matrix_temp_;
  using ACL<T>::d_i_;
  using ACL<T>::l_matrix_;
  using ACL<T>::h_matrix_;
  using ACL<T>::k_matrix_y_;
  using ACL<T>::k_matrix_;
  using ACL<T>::d_matrix_;
  using ACL<T>::d_matrix_to_the_minus_half_;
  using ACL<T>::d_ii_;
  using ACL<T>::didj_matrix_;
  using ACL<T>::gamma_matrix_;
  using ACL<T>::profiler_;
  using ACL<T>::GenDegreeMatrix;
  using ACL<T>::GenKernelMatrix;
  using ACL<T>::OptimizeU;
  using ACL<T>::RunKMeans;
  using ACL<T>::InitYW;
  using ACL<T>::CheckMaxTime;
  using ACL<T>::OutputProgress;
  using ACL<T>::InitXYW;
  using ACL<T>::vectorization_;


  ISM() :
      cost_vector_(),
      eigen_vals_(),
      psi_matrix_(),
      q_matrix_(),
      qt_matrix_(),
      outer_iter_num_(0),
      inner_iter_num_(0)
  {
    method_ = "ISM";
  }

  ~ISM() {}
  ISM(const ISM &rhs) {}

  void GenKernelAndQMatrix(Matrix <T> &input) {
    if (kernel_type_ == kGaussianKernel) {
      float sigma_sq = constant_ * constant_;
      int q_i = 0;
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector <T> delta_ij = input.row(i) - input.row(j);
          T i_j_dist = delta_ij.squaredNorm();
          k_matrix_(i, j) = exp(-i_j_dist / (2 * sigma_sq));
          q_matrix_.row(q_i) = delta_ij;
          q_i++;
        }
      }
      qt_matrix_ = q_matrix_.transpose();
    }
  }
  
  virtual void OutputConfigs() {
    ACL<T>::OutputConfigs();
    std::cout << "vectorizaion: " << vectorization_ << std::endl;
  }

  virtual void InitX(const Matrix <T> &input_matrix) {
    ACL<T>::InitX(input_matrix);
    // Generate Q matrix if using ISM method and vectorization is true
    if (vectorization_) {
      q_matrix_ = Matrix<T>::Zero(n_ * n_, d_);
    }
  }

  void Fit(const Matrix <T> &input_matrix) {
    InitX(input_matrix);
    // When there is no Y, it is the the first round when the second term
    // lambda * HSIC is zero, we do not need to optimize W, and we directly
    // go to kmeans where Y_0 is generated. And both u and v are converged.

    // If this is the first time to generate matrix U
    // then we just use the input X matrix to generate the
    // kernel matrix
    // Because Fit(X) is called when there is no w_matrix generated
    // the kernel matrix is then just the Kernel of X itself
    // If vectorization is enabled,
    // Q matrix is generated along with the Kernel Matrix
    if (vectorization_)
      GenKernelAndQMatrix(x_matrix_);
    else
      GenKernelMatrix(x_matrix_);
    GenDegreeMatrix();
    OptimizeU();
    RunKMeans();
  }

  // Fit() with an empty param list can only be run when the X and Y already
  // exist from the previous round of computation
  void Fit() {
    profiler_["fit"].Start();
    profiler_["exit_timer"].Start();
    PROFILE(InitYW(), profiler_["init"]);
    outer_iter_num_ = 0;
    Vector<T> pre_eigen_vals;
    while (!u_w_converge_ && !max_time_exceeded_ && outer_iter_num_ < 20) {
      if (verbose_)
        std::cout << "\nOuter Loop " << outer_iter_num_ << std::endl;
      pre_u_matrix_ = u_matrix_;
      pre_w_matrix_ = w_matrix_;
      pre_eigen_vals = eigen_vals_;
      // When Fit() is called, we already have a w matrix
      // we project X to subspace W (n * d to d * q)
      // Generate the kernel matrix based on kernel type from projected X
      Matrix <T> projected_matrix = x_matrix_ * w_matrix_;
      // Q matrix is already generated in Fit(X) based on X
      // so there is no need to do it here using the projected_matrix
      GenKernelMatrix(projected_matrix);
      GenDegreeMatrix();
      PROFILE(OptimizeWISM(), profiler_["w"]);
      PROFILE(OptimizeU(), profiler_["u"]);
//        u_converge_ = util::CheckConverged(u_matrix_, pre_u_matrix_,
//                                           threshold2_);
      // we only check if w matrix converges, u matrix does not matter as
      // there are multiple local minimum points
      u_converge_ = true;
      w_converge_ = util::CheckConverged(eigen_vals_,
                                         pre_eigen_vals, threshold1_);
      if (verbose_) {
        T change1 = static_cast<T>((w_matrix_ - pre_w_matrix_).norm()) /
            static_cast<T>(pre_w_matrix_.norm());
        T change2 = static_cast<T>((eigen_vals_ - pre_eigen_vals).norm()) /
            static_cast<T>(pre_eigen_vals.norm());
        std::cout << "change in W | eigen vals: " << change1 << " | " << change2 << std::endl;
      }

      T mean_cost = cost_vector_.mean();
      Vector <T> temp = cost_vector_.array() - mean_cost;
      temp = temp.array() * temp.array();
      T std = sqrt(temp.sum() / temp.rows());
      T min = cost_vector_.minCoeff();
      T max = cost_vector_.maxCoeff();
      T magnitude = fabs(min) > fabs(max) ? fabs(min) : fabs(max);
      w_converge_ |= std < magnitude * 0.01;
      u_w_converge_ = u_converge_ && w_converge_;
      CheckMaxTime();

      if (verbose_)
        OutputProgress();
      outer_iter_num_++;
    }
    PROFILE(RunKMeans(), profiler_["kmeans"]);
    profiler_["fit"].Stop();
    if (outer_iter_num_ >= 20 && verbose_) {
      std::cout << "Reached 20 iterations" << std::endl;
    }
  }

  void Fit(const Matrix <T> &input_matrix, const Matrix <T> &y_matrix) {
    Fit(input_matrix);
    Fit();
  }

  void InitW() {
    // Initialize a d x q zero matrix in ISM mode
    // eigen_vals to bookkeep convergence is also initialized.
    if (w_matrix_.cols() == 0) {
      w_matrix_ = Matrix<T>::Zero(d_, q_);
      eigen_vals_ = Vector<T>::Zero(q_);
    }
  }

  // Initialization for all Y related data structures
  void InitY(const Matrix<T> &y_matrix) {
    ACL<T>::InitY(y_matrix);
  }



  void UpdateW(const Matrix <T> &phi_w) {
    Eigen::EigenSolver <Matrix<T>> solver(phi_w);
    Vector <T> eigen_values = solver.eigenvalues().real();
    Vector <T> eigen_values_img = solver.eigenvalues().imag();
    Vector <T> eigen_vectors_img = solver.eigenvectors().imag();

    // Key-value sort for eigen values
    std::vector <T>
        v(eigen_values.data(), eigen_values.data() + eigen_values.size());
    std::vector <size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t t1, size_t t2) { return v[t1] < v[t2]; });

    for (int i = 0; i < q_; i++) {
      w_matrix_.col(i) = solver.eigenvectors().col(idx[i]).real();
      eigen_vals_(i) = eigen_values(idx[i]);
    }
  }

  virtual void OptimizeWISM(void) {
    profiler_["update_psi"].Start();
    Matrix <T> pre_w_matrix;
    // eigen_vals is the eigen values of matrix Phi(W), used to determine if
    // updating W_{k} has converged
    Vector<T> pre_eigen_vals;
    psi_matrix_ = h_matrix_ * (u_matrix_ * u_matrix_.transpose() -
        k_matrix_y_ * lambda_) * h_matrix_;
    profiler_["update_psi"].Record();
    bool converge = false;
    inner_iter_num_ = 0;
    int max_iter = 10;
    cost_vector_ = Vector<T>::Zero(max_iter);
    while (!converge) {
      pre_w_matrix = w_matrix_;
      pre_eigen_vals = eigen_vals_;
      T objective = 0.0;
      profiler_["update_phi"].Start();
      Matrix <T> phi_w = GenPhiOfW(&objective);
      profiler_["update_phi"].Record();
      cost_vector_(inner_iter_num_) = objective;
      profiler_["update_w"].Start();
      UpdateW(phi_w);
      profiler_["update_w"].Record();

      if (verbose_) {
        std::cout << "Inner Loop " << inner_iter_num_ << ", ";
        std::cout << "Cost: " << objective << ", ";
      }
      if (debug_) {
        std::string out_path =
            "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
        util::ToFile(phi_w, out_path + "phi_w_" + mode_ + "_" + std::to_string(outer_iter_num_) + "_" + std::to_string(inner_iter_num_) + ".csv");
        util::ToFile(w_matrix_, out_path + "w_" + mode_ + "_" + std::to_string(outer_iter_num_) + "_" + std::to_string(inner_iter_num_) + ".csv");
      }

//      converge = util::CheckConverged(w_matrix_, pre_w_matrix, threshold1_);
      converge = util::CheckConverged(eigen_vals_, pre_eigen_vals, threshold1_);


      if (verbose_) {
        T change1 = static_cast<T>((w_matrix_ - pre_w_matrix).norm()) /
            static_cast<T>(pre_w_matrix.norm());
        T change2 = static_cast<T>((eigen_vals_ - pre_eigen_vals).norm()) /
            static_cast<T>(pre_eigen_vals.norm());
        std::cout << "change in W | eigen vals: " << change1 << " | " << change2 << std::endl;
      }

      if (!converge) {
        profiler_["update_k"].Start();
        Matrix <T> projected_matrix = x_matrix_ * w_matrix_;
        GenKernelMatrix(projected_matrix);
        profiler_["update_k"].Record();
        profiler_["update_d"].Start();
        GenDegreeMatrix();
        profiler_["update_d"].Record();
      }


      inner_iter_num_ += 1;
      if (inner_iter_num_ >= max_iter) {
        break;
      }
    }
    profiler_["update_psi"].SumRecords();
    profiler_["update_phi"].SumRecords();
    profiler_["update_w"].SumRecords();
    profiler_["update_k"].SumRecords();
    profiler_["update_d"].SumRecords();

    if (verbose_) {
      if (converge) {
        std::cout << "WISM Converged" << std::endl;
      } else {
        std::cout << "Not converging after " << max_iter
                  << " iterations, but we jump out of the loop anyway\n";
      }
    }
  }

  Matrix <T> GenPhiOfW(T *objective) {
    if (vectorization_) {
      // Vectorization solution, where we convert the conventional for loop
      // solution to matrix multiplications
      float sigma_sq = pow(constant_, 2);
      Matrix <T> ddt = d_i_ * d_i_.transpose();
      Matrix <T> tau = ddt.cwiseProduct(psi_matrix_).cwiseProduct(k_matrix_);
      if (debug_) {
        std::string out_path =
            "/home/xiangyu/Dropbox/git_project/NICE/python/debug/output/";
        util::ToFile(tau, out_path + "tau_" + mode_ + "_"
            + std::to_string(outer_iter_num_) + "_"
            + std::to_string(inner_iter_num_) + ".csv");
      }

      *objective = -tau.sum();
//      Vector<T> test_vector = Vector<T>::Constant(n_*n_, 1);
      Eigen::Map <Vector<T>> tau_map(tau.data(), tau.size());
//      Vector<T> tau_vector = tau_map;
//      qt_matrix_.rowwise() *= test_vector.transpose();
      Matrix <T> phi_w0 = (q_matrix_.array().colwise() * tau_map.array()).matrix().transpose() * q_matrix_;
      Matrix <T> phi_w = phi_w0.array() / sigma_sq;
      return phi_w;
    } else {
      Matrix <T> ddt = d_i_ * d_i_.transpose();
      gamma_matrix_ = ddt.cwiseProduct(psi_matrix_);
      // For loop solution
      float sigma_sq = pow(constant_, 2);
      Matrix <T> phi_w = Matrix<T>::Zero(d_, d_);
      for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
          Vector <T> delta_x_ij =
              this->x_matrix_.row(i) - this->x_matrix_.row(j);
          Matrix <T> a_ij = delta_x_ij * delta_x_ij.transpose();
          Matrix <T> waw = w_matrix_.transpose() * a_ij * w_matrix_;
//          phi_w = phi_w + a_ij * ((gamma_matrix_(i, j) / sigma_sq) *
//              exp(-waw.trace() / (2.0 * sigma_sq)));

          //TODO: Investigate if this should actually be trace()
          T value = gamma_matrix_(i, j) * exp(-waw.trace() / (2.0 * sigma_sq));
          phi_w = phi_w + a_ij * value;
          *objective -= value;
        }
      }
      phi_w /= sigma_sq;
      return phi_w;
    }
  }

 protected:
  // A vector storing every cost in each iteration in WISM
  Vector<T> cost_vector_;
  // Eigen values to determin if W_{k} has converged in ISM
  Vector<T> eigen_vals_;
  // Psi used in vectorized version of ISM
  Matrix <T> psi_matrix_;
  // Q matrix when we do ISM vectorization
  Matrix <T> q_matrix_;
  // Q matrix transpose
  Matrix <T> qt_matrix_;
  // Outer and inner iteration counter for debugging
  int outer_iter_num_;
  int inner_iter_num_;

};
}  // namespace NICE

#endif  // CPP_INCLUDE_ISM_H