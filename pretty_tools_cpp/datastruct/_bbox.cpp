#include"datastruct/datastruct_modules.hpp"
#include<pybind11/numpy.h>
namespace py = pybind11;



py::array_t<float_t> bbox_area(py::array_t<float_t> bboxs)
{

    assert(bboxs.ndim()!=2);

    // box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    py::array_t<float_t> box_area = 
    (bboxs[py::make_tuple(py::ellipsis(),2)] - bboxs[py::make_tuple(py::ellipsis(),0)])*\
    (bboxs[py::make_tuple(py::ellipsis(),3)] - bboxs[py::make_tuple(py::ellipsis(),1)]);

    return box_area;




}



void export_bbox_11(pybind11::module&& m)
{
    m.doc() = "bbox module";
    m.def("bbox_area",&bbox_area,py::arg("bboxs"));
}
