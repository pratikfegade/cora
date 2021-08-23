#ifndef TVM_TE_DIMENSION_H_H_
#define TVM_TE_DIMENSION_H_H_

#include <tvm/ir/attrs.h>
#include <tvm/node/container.h>
#include <tvm/node/node.h>
#include <tvm/runtime/object.h>

#include <unordered_map>

namespace tvm {
namespace te {
class Dimension;

class DimensionNode : public runtime::Object {
 public:
  enum DimensionType : int { kScanDim = 0, kRangeDim = 1, kFunDim = 2 };

  std::string name;
  DimensionType type;

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  TVM_DLL static Dimension make(std::string name, DimensionNode::DimensionType type);

  TVM_DLL bool isFunDim() const;

  TVM_DLL bool isRangeDim() const;

  TVM_DLL bool isScanDim() const;

  TVM_DLL bool isLoopDim() const;

  static constexpr const char* _type_key = "te.Dimension";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimensionNode, Object);
};

struct DimKey {
  enum OpType : int { kFuse = 0, kSplitOuter = 1, kSplitInner = 2, kRebase = 3 };

  const OpType op;
  const DimensionNode* dim1;
  const DimensionNode* dim2;
  const ObjectRef dim1_min_uf;
  const ObjectRef dim2_min_uf;
  const ObjectRef dim1_ext_uf;
  const ObjectRef dim2_ext_uf;

  static inline DimKey FuseKey(const ObjectRef& dim1, const ObjectRef& dim2,
                               const ObjectRef& dim1_min_uf, const ObjectRef& dim2_min_uf,
                               const ObjectRef& dim1_ext_uf, const ObjectRef& dim2_ext_uf) {
    return {kFuse,
            static_cast<const DimensionNode*>(dim1.get()),
            static_cast<const DimensionNode*>(dim2.get()),
            dim1_min_uf,
            dim2_min_uf,
            dim1_ext_uf,
            dim2_ext_uf};
  }
  static inline DimKey SplitOuterKey(const ObjectRef& dim1, const ObjectRef& dim1_min_uf,
                                     const ObjectRef& dim1_ext_uf) {
    return {kSplitOuter,
            static_cast<const DimensionNode*>(dim1.get()),
            nullptr,
            dim1_min_uf,
            dim1_ext_uf,
            NullValue<ObjectRef>(),
            NullValue<ObjectRef>()};
  }
  static inline DimKey SplitInnerKey(const ObjectRef& dim1, const ObjectRef& dim1_min_uf,
                                     const ObjectRef& dim1_ext_uf) {
    return {kSplitInner,
            static_cast<const DimensionNode*>(dim1.get()),
            nullptr,
            dim1_min_uf,
            dim1_ext_uf,
            NullValue<ObjectRef>(),
            NullValue<ObjectRef>()};
  }
  static inline DimKey RebaseKey(const ObjectRef& dim1, const ObjectRef& dim1_min_uf,
                                 const ObjectRef& dim1_ext_uf) {
    return {kRebase,
            static_cast<const DimensionNode*>(dim1.get()),
            nullptr,
            dim1_min_uf,
            dim1_ext_uf,
            NullValue<ObjectRef>(),
            NullValue<ObjectRef>()};
  }
};

class DimKeyHasher {
 public:
  size_t operator()(DimKey d) const;
};
class DimKeyEquality {
 public:
  bool operator()(DimKey d1, DimKey d2) const;
};

class Dimension : public runtime::ObjectRef {
 public:
  static std::unordered_map<DimKey, const DimensionNode*, DimKeyHasher, DimKeyEquality> op_dim_map;

  static Dimension get_or_create_dimension(const DimKey& key);

  Dimension() {}
  // construct from shared ptr.
  explicit Dimension(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const DimensionNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = DimensionNode;
};

inline const DimensionNode* Dimension::operator->() const {
  return static_cast<const DimensionNode*>(data_.get());
}
}  // namespace te
}  // namespace tvm

#endif
