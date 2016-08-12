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

//----------------------------------------------------------------------
//  Programmer: David Mount
//  Last modified:  05/14/04
//  Description:  Sample program for kmeans
//----------------------------------------------------------------------
// Copyright (C) 2004-2005 David M. Mount and University of Maryland
// All Rights Reserved.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.  See the file Copyright.txt in the
// main directory.
//
// The University of Maryland and the authors make no representations
// about the suitability or fitness of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//----------------------------------------------------------------------

#ifndef CPP_INCLUDE_KMEANS_H_
#define CPP_INCLUDE_KMEANS_H_

#include <iostream>
#include <string>
#include "KMeans.h"
#include "KMfilterCenters.h"
#include "KMlocal.h"
#include "include/matrix.h"
#include "include/vector.h"

namespace Nice {

// This class implements the Kmeans function from the Kmlocal library
class Kmeans {
 public:
  enum cluster_type {  // This enum defines the possible cluster types
    lloyds,
    swap,
    simple_hybrid,
    hybrid
  };

  // This enum defines the possible distance calculation types for Fit
  enum distance_type {
    euclidean,
    manhattan,
    cosine
  };

  // This enum is used for parameters to the total number of stages
  enum {
    KM_TERM_CONST,  // constant term
    KM_TERM_LIN_K,  // linear k multiplier
    KM_TERM_LIN_N,  // linear n multiplier
    KM_TERM_POW,  // power exponent
    KM_TERM_VEC_LEN  // length of termination param vector
  };

  /// This is a constructor that assigns default values to member variables to
  /// a Kmeans object
  Kmeans() {
    dims_ = 0;    // dimension
    num_pts_ = 40;    // max number of data points
    k_ = 4;    // number of centers
    cluster_type_ = lloyds;  // Type of clustering algorithm used
    dist_type_ = manhattan;  // Type of distance calculation used
    ctrs_data_ = NULL;  // Used in calculation of the cluster centers
    data_pts_ = NULL;  // Is a member of the ctrs_data_ object
    // The following are default values for the clustering algorithms
    // The maximum number of stages is calculated by
    // MAX_STAGE = (const) + ((lin_k)*k + (lin_n)*n)^(pow)
    max_tot_stage_vec_[KM_TERM_CONST] = 100;
    max_tot_stage_vec_[KM_TERM_LIN_K] = 0;
    max_tot_stage_vec_[KM_TERM_LIN_N] = 0;
    max_tot_stage_vec_[KM_TERM_POW] = 0;
    min_consec_rdl_ = .10;      // min consecutive RDL
    min_accum_rdl_ = .10;     // min accumulated RDL
    max_run_stage_ = 3;     // max stages/run for Lloyd's
    init_prob_accept_ = .5;      // initial prob. of acceptance
    temp_run_length_ = 10;     // length of temp run
    temp_reduc_fact_ = 0.95;     // temperature reduction factor
  }

  /// This function calculates the cluster centers of a given set of points
  /// And stores them in a private member of the Kmeans class
  void Fit(const Nice::Matrix<double> &in_matrix) {
    dims_ = in_matrix.cols();    // dimension
    num_pts_ = in_matrix.rows();    // max number of data points
    data_pts_ = new KMdata(dims_, num_pts_);    // allocate data storage
    KMterm km_specs(max_tot_stage_vec_[KM_TERM_CONST],
                    max_tot_stage_vec_[KM_TERM_LIN_K],
                    max_tot_stage_vec_[KM_TERM_LIN_N],
                    max_tot_stage_vec_[KM_TERM_POW],  // run for 100 stages
        min_consec_rdl_,     // min consec RDL
        min_accum_rdl_,     // min accum RDL
        max_run_stage_,      // max run stages
        init_prob_accept_,     // init. prob. of acceptance
        temp_run_length_,     // temp. run length
        temp_reduc_fact_);      // temp. reduction factor
    for (int pt = 0; pt < num_pts_; ++pt)
      Eigen::Map<Nice::Matrix<double>>((*data_pts_)[pt], 1, dims_) = in_matrix
          .row(pt);  // Map function copies data by row
    data_pts_->setNPts(num_pts_);      // set number of pts
    data_pts_->buildKcTree();      // build filtering structure
    ctrs_data_ = new KMfilterCenters(k_, (*data_pts_));   // allocate centers
    switch (cluster_type_) {
      case lloyds: {
        KMlocalLloyds km_lloyds((*ctrs_data_), km_specs);   // Lloyds
        (*ctrs_data_) = km_lloyds.execute();
        break;
      }
      case swap: {
        KMlocalSwap km_swap((*ctrs_data_), km_specs);   // Swap
        (*ctrs_data_) = km_swap.execute();
        break;
      }
      case simple_hybrid: {
        KMlocalEZ_Hybrid km_ez_hybrid((*ctrs_data_), km_specs);  // Simple
        (*ctrs_data_) = km_ez_hybrid.execute();
        break;
      }
      case hybrid: {
        KMlocalHybrid km_hybrid((*ctrs_data_), km_specs);   // Hybrid
        (*ctrs_data_) = km_hybrid.execute();
        break;
      }
      default:
        break;
    }
  }

  /// This function calculates the K-Means centers and the cluster
  /// assignments for each data point.
  ///
  /// \return
  /// This function returns a pointer to a Vector with the cluster assignments
  Nice::Vector<int> GetLabels() {
    KMctrIdxArray close_ctr = new KMctrIdx[num_pts_];
    double* sq_dist = new double[num_pts_];
    ctrs_data_->getAssignments(close_ctr, sq_dist);  // Get data assignments
    // Map function used to copy array to Nice::Vector
    Nice::Vector<int> cluster_labels = Eigen::Map<Nice::Vector<int>>(close_ctr,
                                                                     num_pts_,
                                                                     1);
    delete[] close_ctr;
    delete sq_dist;
    return (cluster_labels);
  }

  /// This function calculates the cluster assignments of a given set of points
  /// based on previously calculated centers from the Fit() function
  ///
  /// \param in_matrix
  /// Input Matrix passed by reference
  ///
  /// \return
  /// This function returns a Nice Vector with the cluster assignments
  Nice::Vector<int> Predict(const Nice::Matrix<double> &in_matrix) {
    num_pts_ = in_matrix.rows();  // Assign number of points
    double ** centers = ctrs_data_->getCtrPts();  // Holds centers in 2-D array
    // Holds distance to each cluster center
    double * distance_to_centers = new double[k_];
    // Holds closest center for each point
    int * closest_centers = new int[num_pts_];
    // Distance used to calculate distance to a cluster center
    double distance = 0;
    if (dist_type_ == manhattan) {  // If using manhattan distance
      for (int points = 0; points < in_matrix.rows(); ++points) {  // Per point
        for (int center = 0; center < k_; ++center) {  // Per center
          for (int dim = 0; dim < dims_; ++dim) {  // Per dimension
            // Manhattan distance calculation
            distance += abs(centers[center][dim] - in_matrix(points, dim));
          }
          distance_to_centers[center] = distance;  // Store distance value
          distance = 0;  // Reset distance
        }
        // Iterate through distances to find and assign minimum
        int center_min_dist = INT_MAX;  // Set minimum value to max
        for (int center = 0; center < k_; ++center) {  // Per center
          if (distance_to_centers[center] < center_min_dist) {  // Smaller dist
            closest_centers[points] = center;  // Assign center
            center_min_dist = distance_to_centers[center];  // Assign minimum
          }
        }
      }
    } else {
      std::cerr << "Error, this distance type is not defined!" << std::endl;
    }
    Nice::Vector<int> cluster_labels = Eigen::Map<Nice::Vector<int>>(
        closest_centers, num_pts_, 1);  // Map function used to copy array
    delete distance_to_centers;  // Free memory
    delete closest_centers;
    return cluster_labels;
  }

/// This destructor frees memory allocations created within the kmeans class
  ~Kmeans() {
    delete data_pts_;
    delete ctrs_data_;
  }

  /// This function sets the number of clusters used in K-means
  ///
  /// \param
  /// Number of clusters
  void SetK(int num_clusters) {
    k_ = num_clusters;
  }

  /// This function returns the number of clusters used in K-means
  ///
  /// \return
  /// Number of clusters
  int GetK() {
    return k_;
  }

  /// This function sets the number of clusters used in K-means
  void SetClusterType(cluster_type clustering_method) {
    cluster_type_ = clustering_method;
  }

  /// This function returns the clustering type
  ///
  /// \return
  /// Type of clustering
  string GetClusterType() {
    switch (cluster_type_) {
      case (lloyds):  // Using enum definition
        return "Lloyds";
      case (swap):
        return "Swap";
      case (simple_hybrid):
        return "Simple Hybrid";
      case (hybrid):
        return "Hybrid";
      default:
        return "Cluster type error";
    }
  }

  /// This function sets the max number of stages used in K-means
  ///
  /// \param
  /// Number of clusters
  void SetMaxTotStage(int index, double val) {  // set max stage parameters
    assert(index >= 0 && index < KM_TERM_VEC_LEN);
    max_tot_stage_vec_[index] = val;
  }

  /// This function returns the maximum number of stages
  ///
  /// \return
  /// Number of stages
  int GetMaxTotStage() {
    double count = max_tot_stage_vec_[KM_TERM_CONST];
    if (max_tot_stage_vec_[KM_TERM_POW] != 0) {
      double sum = max_tot_stage_vec_[KM_TERM_LIN_K] * k_
          + max_tot_stage_vec_[KM_TERM_LIN_N] * num_pts_;
      count += pow(sum, max_tot_stage_vec_[KM_TERM_POW]);
    }
    assert(count >= 0 && count <= INT_MAX);  // should be positive integer
    if (count <= 0)
      count = INT_MAX;    // 0 means infinity
    return static_cast<int>(count);
  }

  /// This function sets the absolute max number of stages used in K-means
  ///
  /// \param
  /// Number of clusters
  void SetAbsMaxTotStage(int stages) {   // set max number of stages
    max_tot_stage_vec_[KM_TERM_CONST] = stages;
    max_tot_stage_vec_[KM_TERM_POW] = 0;
  }

  /// This function sets the minimum consecutive RDL
  ///
  /// \param
  /// Minimum consecutive RDL
  void SetMinConsecRDL(double rdl) {
    min_consec_rdl_ = rdl;
  }

  /// This function returns the minimum consecutive RDL
  ///
  /// \return
  /// Minumum consecutive RDL
  double GetMinConsecRDL() {
    return min_consec_rdl_;
  }

  /// This function sets the minimum accumulated RDL
  ///
  /// \param
  /// Minimum accumulated RDL
  void SetMinAccumRDL(double rdl) {
    min_accum_rdl_ = rdl;
  }

  /// This function returns the minimum accumulated RDL
  ///
  /// \return
  /// Minumum accumulted RDL
  double GetMinAccumRDL() {
    return min_accum_rdl_;
  }

  /// This function sets the max number of runs per stage used in K-means
  ///
  /// \param
  /// Number of runs per stage
  void SetMaxRunStage(int ms) {
    max_run_stage_ = ms;
  }

  /// This function returns the maximum number of runs per stage
  ///
  /// \return
  /// Maximum runs per stage
  int GetMaxRunStage() {
    return max_run_stage_;
  }

  /// This function sets the initial probability of acceptance
  ///
  /// \param
  /// Initial probability of acceptance
  void SetInitProbAccept(double ipa) {
    init_prob_accept_ = ipa;
  }

  /// This function returns the initial probability of acceptance
  ///
  /// \return
  /// Initial probability of acceptance
  double GetInitProbAccept() const {
    return init_prob_accept_;
  }

  /// This function sets the temperature run length
  ///
  /// \param
  /// Temperature run length
  void SetTempRunLength(int trl) {
    temp_run_length_ = trl;
  }

  /// This function returns the temperature run length
  ///
  /// \return
  /// Temperature run length
  int GetTempRunLength() const {
    return temp_run_length_;
  }

  /// This function sets the temperature reduction factor
  ///
  /// \param
  /// Temperature reduction factor
  void SetTempReducFact(double trf) {
    temp_reduc_fact_ = trf;
  }

  /// This function returns the temperature reduction factor
  ///
  /// \return
  /// Temperature reduction factor
  double GetTempReducFact() {
    return temp_reduc_fact_;
  }

 private:
  int k_;  // Number of clusters
  int dims_;  // Dimensions
  int num_pts_;  // Number of points
  KMfilterCenters* ctrs_data_;  // Stores cluster center data
  KMdata* data_pts_;  // Stored because it's part of the KMfilterCenters object
  cluster_type cluster_type_;  // Type of clustering algorithm
  distance_type dist_type_;  // Type of distance calculation
  double max_tot_stage_vec_[KM_TERM_VEC_LEN];  // max total stages
  double min_consec_rdl_;      // min consecutive RDL
  double min_accum_rdl_;     // min accumulated RDL
  int max_run_stage_;     // max stages/run for Lloyd's
  double init_prob_accept_;      // initial prob. of acceptance
  int temp_run_length_;     // length of temp run
  double temp_reduc_fact_;     // temperature reduction factor
};

}  // namespace Nice
#endif  // CPP_INCLUDE_KMEANS_H_
