#ifndef TVM_TE_DIMENSION_RELATIONS_H_H_
#define TVM_TE_DIMENSION_RELATIONS_H_H_

#include <tvm/runtime/object.h>
#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <unordered_map>

namespace tvm {
namespace te {
class DimensionRelationNode;

/*!
 * \brief The schedule relation between Dimensions
 *  can be Split, Fuse.
 */
class DimensionRelation : public ObjectRef {
 public:
  DimensionRelation() {}
  explicit DimensionRelation(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const DimensionRelationNode* operator->() const;
};

/*! \brief base node of iteration var */
 class DimensionRelationNode : public Object {
 public:
  static constexpr const char* _type_key = "DimensionRelation";
  TVM_DECLARE_BASE_OBJECT_INFO(DimensionRelationNode, Object);
};

/*!
 * \brief Split the parent domain into product of
 *  outer and iter.
 */
class DimensionSplitNode : public DimensionRelationNode {
 public:
  /*! \brief The parent domain */
  Dimension parent;
  /*! \brief The outer domain */
  Dimension outer;
  /*! \brief The inner domain */
  Dimension inner;
  /*! \brief The split factor */
  PrimExpr factor;
  /*! \brief Number of parts, only factor or nparts can be given */
  PrimExpr nparts;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("factor", &factor);
    v->Visit("nparts", &nparts);
  }

  static DimensionRelation make(Dimension parent,
				Dimension outer,
				Dimension inner,
				PrimExpr factor,
				PrimExpr nparts);

  static constexpr const char* _type_key = "DimensionSplit";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimensionSplitNode, DimensionRelationNode);
};

/*!
 * \brief Fuse two domains into one domain.
 */
class DimensionFuseNode : public DimensionRelationNode {
 public:
  /*! \brief The outer domain */
  Dimension outer;
  /*! \brief The inner domain */
  Dimension inner;
  /*! \brief The target domain */
  Dimension fused;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("fused", &fused);
  }

  static DimensionRelation make(
      Dimension outer, Dimension inner, Dimension fused);

  static constexpr const char* _type_key = "DimensionFuse";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimensionFuseNode, DimensionRelationNode);
};

class DimensionRelationGraphNode;

/*!
 * \brief The schedule relation between Dimensions
 *  can be Split, Fuse.
 */
class DimensionRelationGraph : public ObjectRef {
 public:
  DimensionRelationGraph() {}
  explicit DimensionRelationGraph(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  // inline const DimensionRelationGraphNode* operator->() const;
  inline DimensionRelationGraphNode* operator->() const;
};

/*! \brief base node of iteration var */
class DimensionRelationGraphNode : public Object {
public:
  Array<DimensionRelation> relations;
  Array<Dimension> leaf_dimensions;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("relations", &relations);
    v->Visit("leaf_dimensions", &leaf_dimensions);
  }

  static DimensionRelationGraph make(Array<Dimension> root_dimensions);

  static constexpr const char* _type_key = "DimensionRelationGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimensionRelationGraphNode, Object);
};

// implementations
inline const DimensionRelationNode* DimensionRelation::operator->() const {
  return static_cast<const DimensionRelationNode*>(get());
}
// inline const DimensionRelationGraphNode* DimensionRelationGraph::operator->() const {
  // return static_cast<const DimensionRelationGraphNode*>(get());
// }
inline DimensionRelationGraphNode* DimensionRelationGraph::operator->() const {
  return const_cast<DimensionRelationGraphNode*>(static_cast<const DimensionRelationGraphNode*>(get()));
}
}
}

#endif
