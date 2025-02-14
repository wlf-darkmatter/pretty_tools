#include"datastruct/datastruct_modules.hpp"
#include<pybind11/numpy.h>
namespace py = pybind11;



// 适用模板，可以绑定多种类型 比如int uint int64 uint64
// TODO 这个不行，编译报错
// template<typename dtype>
// py::tuple cy_get_gt_match_from_id(py::array_t<dtype>& np_id_a,
//                                         py::array_t<dtype>& np_id_b
// )
#define dtype int64_t
py::tuple cy_get_gt_match_from_id(py::array_t<dtype>& np_id_a,
                                        py::array_t<dtype>& np_id_b
)
{
    int i,j;
    uint32_t N = np_id_a.shape(0);
    uint32_t M = np_id_b.shape(0);
    uint32_t K = std::max(N,M);
    uint32_t count = 0;

    auto np = py::module::import("numpy");
    py::print("data type is ",np_id_a.attr("dtype"));
    py::array_t<dtype> matched = np.attr("zeros")(py::make_tuple(K,3),np_id_a.attr("dtype"));

    // 这个提供一个代理，可以访问array的数据，但是不能修改，且不做边界检查（操作不当会越界）
    // 想要能修改，可以使用mutable_unchecked
    // 这种是已知数据纬度的，速度会比较快
    // 不知道时，手动通过datatype、指针和stride去找
    auto p_np_id_a = np_id_a.unchecked<1>();
    auto p_np_id_b = np_id_b.unchecked<1>();
    auto p_matched = matched.mutable_unchecked<2>();

    for(size_t i=0;i<N;i++)
    {
        for(size_t j=0;j<M;j++)
        {
            if(p_np_id_a[i] == p_np_id_b[j])
            {
                p_matched(count,0) = i;
                p_matched(count,1) = j;
                p_matched(count,2) = p_np_id_a[i];
                count++;
                continue;
            }
        }
    }
    py::slice slice(0,count,1);

    // matched[0:count,0:2]
    // matched[py::make_tuple(py::slice(0,count,1),py::slice(0,2,1))];
    // matched[:count, 2]
    // matched[py::make_tuple(py::slice(0,count,1),2)]

    return py::make_tuple(matched[py::make_tuple(py::slice(0,count,1),py::slice(0,2,1))],
                          matched[py::make_tuple(py::slice(0,count,1),2)]);
}



// 导出
void export_misc_11(py::module&& m){
    // m.def_submodule("")
    // m.def("cy_get_gt_match_from_id",&cy_get_gt_match_from_id);

    // auto submodule = m.def_submodule("_C_misc_11", "submodule for datastruct");
    // submodule.def("cy_get_gt_match_from_id",&cy_get_gt_match_from_id);

    m.doc() = "misc_11 module";


    // 返回这块，还有不同策略，可以看看官网
    // 编译不过
#ifdef dtype
    m.def("cy_get_gt_match_from_id",&cy_get_gt_match_from_id);
#else
    m.def("cy_get_gt_match_from_id",&cy_get_gt_match_from_id<int64_t>);
    m.def("cy_get_gt_match_from_id",&cy_get_gt_match_from_id<int32_t>);
#endif
    // m.def("cy_get_gt_match_from_id",[](py::array_t<int64_t> np_id_a,
    //                                     py::array_t<int64_t> np_id_b){
    //     return cy_get_gt_match_from_id<int64_t>(np_id_a,np_id_b);});
    
    // m.def("cy_get_gt_match_from_id",[](py::array_t<int32_t> np_id_a,
    //                                     py::array_t<int32_t> np_id_b){
    //     return cy_get_gt_match_from_id<int32_t>(np_id_a,np_id_b);});

}