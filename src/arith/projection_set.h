/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file int_set.h
 *
 * \brief Very simple and conservative implementation of an internal
 * data structure for set representing the projection of an
 * uninterpreted function, given the values (in the form of IntSets)
 * of it's arguments.
 */
#ifndef TVM_ARITH_PROJECTION_SET_H_
#define TVM_ARITH_PROJECTION_SET_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/uninterp_fun.h>
#include <limits>
#include "const_fold.h"

namespace tvm {
namespace arith {

using tir::UninterpFun;

/*!
 * \brief Symbolic projection set.
 *
 * \note We intentionally keep the internal of IntSet private,
         as we might change it later.
 */
class ProjectionSetNode : public IntSetNode {
 public:
  /*! \brief Uninterpreted function. */
  UninterpFun ufun;
  /*! \brief Values of the arguments to the function. */
  Map<te::Dimension, IntSet> arguments;

  // visitor overload.
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ufun", &ufun);
    v->Visit("arguments", &arguments);
  }

  /*! \return Whether the interval has upper bound. */
  bool HasUpperBound() const {
    return false;
  }
  /*! \return Whether the interval has lower bound. */
  bool HasLowerBound() const {
    return false;
  }
  /*! \return Whether the interval is a single point. */
  bool IsSinglePoint() const {
    return false;
  }
  /*! \return whether interval represent nothing */
  bool IsEmpty() const {
    // during computations, either extreme could occur.
    return false;
  }
  /*! \return whether interval represent everything */
  bool IsEverything() const {
    return false;
  }

  static constexpr const char* _type_key = "arith.ProjectionSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProjectionSetNode, IntSetNode);
};

/*!
 * \brief Interval set used for symbolic integer analysis.
 * \sa ProjectionSetNode
 */
class ProjectionSet : public IntSet {
 public:
  /*!
   * \brief Make a new instance of interval set.
   * \param min_value The minimum value in the interval.
   * \param max_value The maximum value in the interval.
   * \return The created set.
   */
  TVM_DLL ProjectionSet(UninterpFun ufun, Map<te::Dimension, IntSet> arguments);

  TVM_DEFINE_OBJECT_REF_COW_METHOD(ProjectionSetNode);
  TVM_DEFINE_OBJECT_REF_METHODS(ProjectionSet, IntSet, ProjectionSetNode);
};

/* /\*! */
/*  * \brief Create union of two ProjectionSets. */
/*  * \param analyzer The analyzer for simplification analysis. */
/*  * \param a The first set. */
/*  * \param b The second set. */
/*  * \return The result set. */
/*  *\/ */
/* TVM_DLL ProjectionSet Union(Analyzer* analyzer, ProjectionSet a, ProjectionSet b); */

/* /\*! */
/*  * \brief Create insersection of two ProjectionSets. */
/*  * \param analzyer The analyzer for simplification analysis. */
/*  * \param a The first set. */
/*  * \param b The second set. */
/*  * \return The result set. */
/*  *\/ */
/* TVM_DLL ProjectionSet Intersect(Analyzer *analzyer, ProjectionSet a, ProjectionSet b); */

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_PROJECTION_SET_H_
