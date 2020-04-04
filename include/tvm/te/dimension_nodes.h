// #ifndef TVM_TE_DIMENSION_NODES_H_H_
// #define TVM_TE_DIMENSION_NODES_H_H_

// #include <tvm/ir/expr.h>
// #include <tvm/tir/expr.h>
// #include <tvm/tir/uninterp_fun.h>
// // #include <tvm/te/dimension.h>
// #include <unordered_map>

// namespace tvm {
//   namespace te {
//     class Dimension;

//     class DimensionNode : public Object {
//     public:
//       enum DimensionType: int {
//         kRangeDim = 0,
//         kFunDim = 1
//       };

//       std::string name;
//       DimensionType type;

//       void VisitAttrs(AttrVisitor* v) {
// 	v->Visit("name", &name);
//       }

//       TVM_DLL static Dimension make(std::string name, DimensionNode::DimensionType type);

//       static constexpr const char* _type_key = "te.Dimension";
//       TVM_DECLARE_FINAL_OBJECT_INFO(DimensionNode, Object);
//     };
//   }
// }

// #endif
