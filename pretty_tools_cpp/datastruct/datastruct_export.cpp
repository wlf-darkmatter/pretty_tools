#include"datastruct/datastruct_modules.hpp"

void bind_datastruct_module(pybind11::module&& m)
{
    m.doc() = "c++ impl datastruct";
    // 这个模块下可能由多个子模块，可以类似上边的，为每个子模块继续创建用于初始化的函数，这边负责调用
    export_misc_11(m.def_submodule("misc_11"));
    export_bbox_11(m.def_submodule("bbox_11"));
}
