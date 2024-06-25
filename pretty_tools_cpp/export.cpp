#include"datastruct/datastruct_modules.hpp"

// setup时，Pybind11Extension的name写为pretty_tools._C_pretty_tools
// 这样，pip install .时，pyd就在pretty_tools下

PYBIND11_MODULE(_C_pretty_tools, m){
    m.doc() = "pretty_tools cpp bindings";

    // 添加子模块 bind_datastruct_module在子模块中定义并实现
    bind_datastruct_module(m.def_submodule("datastruct"));
}
