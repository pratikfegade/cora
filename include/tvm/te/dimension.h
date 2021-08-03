#ifndef TVM_TE_DIMENSION_H_H_
#define TVM_TE_DIMENSION_H_H_

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
};

class DimKeyHasher {
 public:
  size_t operator()(DimKey d) const {
    auto ptr_hash = std::hash<const DimensionNode*>();
    return ptr_hash(d.dim1) ^ ptr_hash(d.dim2) ^ std::hash<DimKey::OpType>()(d.op);
  }
};

class DimKeyEquality {
 public:
  bool operator()(DimKey d1, DimKey d2) const {
    return d1.dim1 == d2.dim1 && d1.dim2 == d2.dim2 && d1.op == d2.op;
  }
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
