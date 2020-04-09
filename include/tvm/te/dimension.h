#ifndef TVM_TE_DIMENSION_H_H_
#define TVM_TE_DIMENSION_H_H_

#include <tvm/runtime/object.h>
#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <unordered_map>

namespace tvm {
  namespace te {
    class Dimension;

    class DimensionNode : public runtime::Object {
    public:
      enum DimensionType: int {
        kRangeDim = 0,
        kFunDim = 1
      };

      std::string name;
      DimensionType type;

      void VisitAttrs(AttrVisitor* v) {
	v->Visit("name", &name);
      }

      TVM_DLL static Dimension make(std::string name, DimensionNode::DimensionType type);

      static constexpr const char* _type_key = "te.Dimension";
      TVM_DECLARE_FINAL_OBJECT_INFO(DimensionNode, Object);
    };

    class Dimension : public runtime::ObjectRef {
    public:
      static Dimension NoDimension;

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
  }
}

#endif
