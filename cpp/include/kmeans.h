// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CPP_INCLUDE_KMEANS_H_
#define CPP_INCLUDE_KMEANS_H_

#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include "include/matrix.h"
#include "include/vector.h"


namespace Nice {


template<typename T>
class KMeans {
 public:
//  void Fit1(const Matrix<T> &input_data, int k) {
//    int num_features = input_data.cols();
//    int num_samples = input_data.rows();
//    typedef dlib::matrix<T> sample_type;
//    typedef dlib::radial_basis_kernel<sample_type> kernel_type;
//    std::vector<sample_type> samples;
//    std::vector<sample_type> initial_centers;
//    sample_type m;
//    m.set_size(num_features, 1);
//    for (long i = 0; i < num_samples; i++) {
//      for (long j = 0; j < num_features; j++)
//        m(j) = input_data(i, j);
//      samples.push_back(m);
//    }
//    dlib::kcentroid<kernel_type> kc(kernel_type(0.01), 0.0001, 20);
//    dlib::kkmeans<kernel_type> km(kc);
//    km.set_number_of_centers(k);
//    dlib::pick_initial_centers(k, initial_centers, samples, km.get_kernel());
//    km.train(samples, initial_centers);
////    Vector<T> assignments(num_samples);
//    labels_ = Vector<T>::Zero(num_samples);
//    for (long i = 0; i < num_samples; i++) {
////      std::cout << samples[i] << std::endl;
//      labels_(i) = km(samples[i]);
//    }
//  }

  void Fit(const Matrix<T> &input_data, int k) {
    k_ = k;
    centers_.resize(input_data.cols(), k_);
    Run(input_data.transpose());
  }

  void Run(const Matrix<T> &input_data) {
    if (input_data.cols() < k_) {
      std::stringstream ss;
      ss << "The number of points (" << input_data.cols()
         << ") must be larger than the number of clusters (" << k_ << ")";
      throw std::runtime_error(ss.str());
    }
    // Seed a random number generator
    if (random_) {
      unsigned int t = time(NULL);
      srand48(t);
    } else {
      srand48(0);
    }
    KMeansPPInit(input_data);

    // We must store the labels at the previous iteration to
    // determine whether any labels changed at each iteration.
    // initialize to all zeros
    Vector<T> old_labels = Vector<T>::Zero(input_data.cols());

    // Initialize the labels array
    labels_.resize(input_data.cols());
    // The current iteration number
    int iter = 0;

    // Track whether any labels changed in the last iteration
    bool changed = true;
    do {
      AssignLabels(input_data);
      EstimateNewCenters(input_data);
      changed = CheckChanged(labels_, old_labels);
      // Save the old labels
      old_labels = labels_;
      iter++;
    } while (changed);
  }
  unsigned int FindClosestCluster(const Vector<T>& query_point,
                                  unsigned int num_cluster) {
    unsigned int closest_cluster = 0;
    T min_dist = std::numeric_limits<T>::max();
    for (unsigned int i = 0; i < num_cluster; ++i) {
      T dist = (centers_.col(i) - query_point).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_cluster = i;
      }
    }
    return closest_cluster;
  }
  void AssignLabels(const Matrix<T> &input_data) {
    // Assign each point to the closest cluster
    for (unsigned int point = 0; point < input_data.cols(); ++point) {
      unsigned int closest_cluster =
          FindClosestCluster(input_data.col(point), k_);
      labels_(point) = (T)closest_cluster;
    }
  }
  std::vector<unsigned int> GetIndicesWithLabel(const T label) const {
    std::vector<unsigned int> indices_with_label;
    for (unsigned int i = 0; i < labels_.size(); i++) {
      if (labels_(i) == label) {
        indices_with_label.push_back(i);
      }
    }
    return indices_with_label;
  }
  void EstimateNewCenters(const Matrix<T> &input_data) {
    Matrix<T> old_centers = centers_;
    for (unsigned int cluster = 0; cluster < k_; ++cluster) {
      std::vector<unsigned int> indices_with_label =
          GetIndicesWithLabel(cluster);
      Matrix<T> cluster_points(input_data.rows(), indices_with_label.size());
      for (unsigned int point = 0; point < indices_with_label.size(); point++) {
        cluster_points.col(point) = input_data.col(indices_with_label[point]);
      }
      Vector<T> center;
      if (cluster_points.cols() == 0) {
        center = old_centers.col(cluster);
      } else {
        center = cluster_points.rowwise().mean();
      }
      centers_.col(cluster) = center;
    }
  }
  bool CheckChanged(const Vector<T>& labels, const Vector<T>& old_labels) {
    bool changed = false;
    for (unsigned int i = 0; i < labels.size(); i++) {
      if (labels(i) != old_labels(i)) {
        changed = true;
        break;
      }
    }
    return changed;
  }

  unsigned int SelectWeightedIndex(Vector<T> weights) {
    // Normalize
    Vector<T> normalizedWeights = weights / weights.sum();
    std::map<T, unsigned int> weights_index_map;
    for (unsigned int i = 0; i < weights.size(); i++) {
      weights_index_map[normalizedWeights(i)] = i;
    }
    // Sort the normalized weights
    std::sort(normalizedWeights.data(),
      normalizedWeights.data()+normalizedWeights.size());
    // Get a randome value between 0 and 1
    T random_value = (T)drand48();

    T running_total = 0.0;

    for (unsigned int i = 0; i < normalizedWeights.size(); i++) {

      running_total += normalizedWeights[i];
      if (random_value < running_total) {
        T weight = normalizedWeights(i);
        return weights_index_map[weight];
      }
    }
    std::cerr << "runningTotal: " << running_total << std::endl;
    std::cerr << "randomValue: " << random_value << std::endl;
    throw std::runtime_error(
      "SelectWeightedIndex() reached end, we should never get here.");
  }

  void KMeansPPInit(const Matrix<T> &input_data) {
    // Assign one center at random
    unsigned int random_id = rand() % input_data.cols();
    Vector<T> p = input_data.col(random_id);
    centers_.col(0) = p;
    // Assign the rest of the initial centers using a
    // weighted probability of the distance to the nearest center
    Vector<T> weights(input_data.cols());
    for (unsigned int cluster = 1; cluster < k_; ++cluster) {
      // Create weight vector
      for (unsigned int i = 0; i < input_data.cols(); i++) {
        Vector<T> current_point = input_data.col(i);
        unsigned int closest_cluster =
            FindClosestCluster(current_point, cluster);
        weights(i) =
            (centers_.col(closest_cluster) - current_point).squaredNorm();
      }
      unsigned int selected_point_id = SelectWeightedIndex(weights);
      p = input_data.col(selected_point_id);
      centers_.col(cluster) = p;
    }
  }

  void SetRandom(const bool r) {
    this->Random = r;
  }

  Matrix <T> GetLabels() {
    return labels_;
  }

  Matrix <T> GetCenters() {
    return centers_;
  }
 private:
  Vector<T> labels_;
  bool random_ = true;
  // The number of clusters to find.
  unsigned int k_;
  // The current cluster centers.
  Matrix<T> centers_;
};
}  // namespace Nice
#endif  // CPP_INCLUDE_KMEANS_H_
