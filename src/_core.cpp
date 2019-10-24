// File: bits/libio.cpp
#include <eddl/utils.h>
#include <sstream> // __str__
#include <stdio.h>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_bits_libio(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // _IO_FILE file:bits/libio.h line:245
		pybind11::class_<_IO_FILE, std::shared_ptr<_IO_FILE>> cl(M(""), "_IO_FILE", "");
		cl.def( pybind11::init( [](){ return new _IO_FILE(); } ) );
		cl.def_readwrite("_flags", &_IO_FILE::_flags);
		cl.def_readwrite("_fileno", &_IO_FILE::_fileno);
		cl.def_readwrite("_flags2", &_IO_FILE::_flags2);
		cl.def_readwrite("_old_offset", &_IO_FILE::_old_offset);
		cl.def_readwrite("_cur_column", &_IO_FILE::_cur_column);
		cl.def_readwrite("_vtable_offset", &_IO_FILE::_vtable_offset);
		cl.def_readwrite("_offset", &_IO_FILE::_offset);
		cl.def_readwrite("__pad5", &_IO_FILE::__pad5);
		cl.def_readwrite("_mode", &_IO_FILE::_mode);
	}
	// get_fmem(int, char *) file:eddl/utils.h line:17
	M("").def("get_fmem", (float * (*)(int, char *)) &get_fmem, "C++: get_fmem(int, char *) --> float *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("str"));

	// humanSize(unsigned long) file:eddl/utils.h line:19
	M("").def("humanSize", (char * (*)(unsigned long)) &humanSize, "C++: humanSize(unsigned long) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("bytes"));

	// get_free_mem() file:eddl/utils.h line:21
	M("").def("get_free_mem", (unsigned long (*)()) &get_free_mem, "C++: get_free_mem() --> unsigned long");

}


// File: eddl/tensor/tensor.cpp
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <tensor_addons.hpp>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tensor file:eddl/tensor/tensor.h line:56
		pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", pybind11::buffer_protocol());
		cl.def( pybind11::init( [](){ return new Tensor(); } ) );
		cl.def( pybind11::init( [](Tensor const &o){ return new Tensor(o); } ) );
		cl.def_readwrite("device", &Tensor::device);
		cl.def_readwrite("ndim", &Tensor::ndim);
		cl.def_readwrite("size", &Tensor::size);
		cl.def_readwrite("shape", &Tensor::shape);
		cl.def_readwrite("stride", &Tensor::stride);
		cl.def_readwrite("gpu_device", &Tensor::gpu_device);
		cl.def("ToCPU", [](Tensor &o) -> void { return o.ToCPU(); }, "");
		cl.def("ToCPU", (void (Tensor::*)(int)) &Tensor::ToCPU, "C++: Tensor::ToCPU(int) --> void", pybind11::arg("dev"));
		cl.def("ToGPU", [](Tensor &o) -> void { return o.ToGPU(); }, "");
		cl.def("ToGPU", (void (Tensor::*)(int)) &Tensor::ToGPU, "C++: Tensor::ToGPU(int) --> void", pybind11::arg("dev"));
		cl.def("clone", (class Tensor * (Tensor::*)()) &Tensor::clone, "C++: Tensor::clone() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("resize", (void (Tensor::*)(int, float *)) &Tensor::resize, "C++: Tensor::resize(int, float *) --> void", pybind11::arg("b"), pybind11::arg("fptr"));
		cl.def("resize", (void (Tensor::*)(int)) &Tensor::resize, "C++: Tensor::resize(int) --> void", pybind11::arg("b"));
		cl.def("resize", (void (Tensor::*)(int, class Tensor *)) &Tensor::resize, "C++: Tensor::resize(int, class Tensor *) --> void", pybind11::arg("b"), pybind11::arg("T"));
		cl.def("isCPU", (int (Tensor::*)()) &Tensor::isCPU, "C++: Tensor::isCPU() --> int");
		cl.def("isGPU", (int (Tensor::*)()) &Tensor::isGPU, "C++: Tensor::isGPU() --> int");
		cl.def("isFPGA", (int (Tensor::*)()) &Tensor::isFPGA, "C++: Tensor::isFPGA() --> int");
		cl.def("info", (void (Tensor::*)()) &Tensor::info, "C++: Tensor::info() --> void");
		cl.def("print", (void (Tensor::*)()) &Tensor::print, "C++: Tensor::print() --> void");
		cl.def("numel", (int (Tensor::*)()) &Tensor::numel, "C++: Tensor::numel() --> int");
		cl.def("set", (void (Tensor::*)(float)) &Tensor::set, "C++: Tensor::set(float) --> void", pybind11::arg("v"));
		cl.def_static("arange", [](float const & a0, float const & a1) -> Tensor * { return Tensor::arange(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"));
		cl.def_static("arange", (class Tensor * (*)(float, float, float, int)) &Tensor::arange, "C++: Tensor::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("range", [](float const & a0, float const & a1) -> Tensor * { return Tensor::range(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"));
		cl.def_static("range", (class Tensor * (*)(float, float, float, int)) &Tensor::range, "C++: Tensor::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("linspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::linspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("linspace", (class Tensor * (*)(float, float, int, int)) &Tensor::linspace, "C++: Tensor::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("logspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::logspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::logspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return Tensor::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
		cl.def_static("logspace", (class Tensor * (*)(float, float, int, float, int)) &Tensor::logspace, "C++: Tensor::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));
		cl.def_static("eye", [](int const & a0) -> Tensor * { return Tensor::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("size"));
		cl.def_static("eye", (class Tensor * (*)(int, int)) &Tensor::eye, "C++: Tensor::eye(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("dev"));
		cl.def("abs_", (void (Tensor::*)()) &Tensor::abs_, "C++: Tensor::abs_() --> void");
		cl.def_static("abs", (class Tensor * (*)(class Tensor *)) &Tensor::abs, "C++: Tensor::abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("acos_", (void (Tensor::*)()) &Tensor::acos_, "C++: Tensor::acos_() --> void");
		cl.def_static("acos", (class Tensor * (*)(class Tensor *)) &Tensor::acos, "C++: Tensor::acos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("add_", (void (Tensor::*)(float)) &Tensor::add_, "C++: Tensor::add_(float) --> void", pybind11::arg("v"));
		cl.def_static("add", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::add, "C++: Tensor::add(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("add", (void (*)(float, class Tensor *, float, class Tensor *, class Tensor *, int)) &Tensor::add, "C++: Tensor::add(float, class Tensor *, float, class Tensor *, class Tensor *, int) --> void", pybind11::arg("scA"), pybind11::arg("A"), pybind11::arg("scB"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("add", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::add, "C++: Tensor::add(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("inc", (void (*)(class Tensor *, class Tensor *)) &Tensor::inc, "C++: Tensor::inc(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("asin_", (void (Tensor::*)()) &Tensor::asin_, "C++: Tensor::asin_() --> void");
		cl.def_static("asin", (class Tensor * (*)(class Tensor *)) &Tensor::asin, "C++: Tensor::asin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("atan_", (void (Tensor::*)()) &Tensor::atan_, "C++: Tensor::atan_() --> void");
		cl.def_static("atan", (class Tensor * (*)(class Tensor *)) &Tensor::atan, "C++: Tensor::atan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("ceil_", (void (Tensor::*)()) &Tensor::ceil_, "C++: Tensor::ceil_() --> void");
		cl.def_static("ceil", (class Tensor * (*)(class Tensor *)) &Tensor::ceil, "C++: Tensor::ceil(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("clamp_", (void (Tensor::*)(float, float)) &Tensor::clamp_, "C++: Tensor::clamp_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("clamp", (class Tensor * (*)(class Tensor *, float, float)) &Tensor::clamp, "C++: Tensor::clamp(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));
		cl.def("clampmax_", (void (Tensor::*)(float)) &Tensor::clampmax_, "C++: Tensor::clampmax_(float) --> void", pybind11::arg("max"));
		cl.def_static("clampmax", (class Tensor * (*)(class Tensor *, float)) &Tensor::clampmax, "C++: Tensor::clampmax(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("max"));
		cl.def("clampmin_", (void (Tensor::*)(float)) &Tensor::clampmin_, "C++: Tensor::clampmin_(float) --> void", pybind11::arg("min"));
		cl.def_static("clampmin", (class Tensor * (*)(class Tensor *, float)) &Tensor::clampmin, "C++: Tensor::clampmin(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"));
		cl.def("cos_", (void (Tensor::*)()) &Tensor::cos_, "C++: Tensor::cos_() --> void");
		cl.def_static("cos", (class Tensor * (*)(class Tensor *)) &Tensor::cos, "C++: Tensor::cos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("cosh_", (void (Tensor::*)()) &Tensor::cosh_, "C++: Tensor::cosh_() --> void");
		cl.def_static("cosh", (class Tensor * (*)(class Tensor *)) &Tensor::cosh, "C++: Tensor::cosh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("inv_", (void (Tensor::*)()) &Tensor::inv_, "C++: Tensor::inv_() --> void");
		cl.def("div_", (void (Tensor::*)(float)) &Tensor::div_, "C++: Tensor::div_(float) --> void", pybind11::arg("v"));
		cl.def_static("div", (class Tensor * (*)(class Tensor *)) &Tensor::div, "C++: Tensor::div(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("el_div", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_div, "C++: Tensor::el_div(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("exp_", (void (Tensor::*)()) &Tensor::exp_, "C++: Tensor::exp_() --> void");
		cl.def_static("exp", (class Tensor * (*)(class Tensor *)) &Tensor::exp, "C++: Tensor::exp(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("floor_", (void (Tensor::*)()) &Tensor::floor_, "C++: Tensor::floor_() --> void");
		cl.def_static("floor", (class Tensor * (*)(class Tensor *)) &Tensor::floor, "C++: Tensor::floor(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log_", (void (Tensor::*)()) &Tensor::log_, "C++: Tensor::log_() --> void");
		cl.def_static("log", (class Tensor * (*)(class Tensor *)) &Tensor::log, "C++: Tensor::log(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log2_", (void (Tensor::*)()) &Tensor::log2_, "C++: Tensor::log2_() --> void");
		cl.def_static("log2", (class Tensor * (*)(class Tensor *)) &Tensor::log2, "C++: Tensor::log2(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log10_", (void (Tensor::*)()) &Tensor::log10_, "C++: Tensor::log10_() --> void");
		cl.def_static("log10", (class Tensor * (*)(class Tensor *)) &Tensor::log10, "C++: Tensor::log10(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("logn_", (void (Tensor::*)(float)) &Tensor::logn_, "C++: Tensor::logn_(float) --> void", pybind11::arg("n"));
		cl.def_static("logn", (class Tensor * (*)(class Tensor *)) &Tensor::logn, "C++: Tensor::logn(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("max", (float (Tensor::*)()) &Tensor::max, "C++: Tensor::max() --> float");
		cl.def("min", (float (Tensor::*)()) &Tensor::min, "C++: Tensor::min() --> float");
		cl.def("mod_", (void (Tensor::*)(float)) &Tensor::mod_, "C++: Tensor::mod_(float) --> void", pybind11::arg("v"));
		cl.def_static("mod", (class Tensor * (*)(class Tensor *, float)) &Tensor::mod, "C++: Tensor::mod(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def("mult_", (void (Tensor::*)(float)) &Tensor::mult_, "C++: Tensor::mult_(float) --> void", pybind11::arg("v"));
		cl.def_static("mult", (class Tensor * (*)(class Tensor *)) &Tensor::mult, "C++: Tensor::mult(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("mult2D", (void (*)(class Tensor *, int, class Tensor *, int, class Tensor *, int)) &Tensor::mult2D, "C++: Tensor::mult2D(class Tensor *, int, class Tensor *, int, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("tA"), pybind11::arg("B"), pybind11::arg("tB"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("el_mult", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_mult, "C++: Tensor::el_mult(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("neg_", (void (Tensor::*)()) &Tensor::neg_, "C++: Tensor::neg_() --> void");
		cl.def_static("neg", (class Tensor * (*)(class Tensor *)) &Tensor::neg, "C++: Tensor::neg(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("normalize_", [](Tensor &o) -> void { return o.normalize_(); }, "");
		cl.def("normalize_", [](Tensor &o, float const & a0) -> void { return o.normalize_(a0); }, "", pybind11::arg("min"));
		cl.def("normalize_", (void (Tensor::*)(float, float)) &Tensor::normalize_, "C++: Tensor::normalize_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("pow_", (void (Tensor::*)(float)) &Tensor::pow_, "C++: Tensor::pow_(float) --> void", pybind11::arg("exp"));
		cl.def_static("pow", (class Tensor * (*)(class Tensor *)) &Tensor::pow, "C++: Tensor::pow(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("reciprocal_", (void (Tensor::*)()) &Tensor::reciprocal_, "C++: Tensor::reciprocal_() --> void");
		cl.def_static("reciprocal", (class Tensor * (*)(class Tensor *)) &Tensor::reciprocal, "C++: Tensor::reciprocal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("remainder_", (void (Tensor::*)(float)) &Tensor::remainder_, "C++: Tensor::remainder_(float) --> void", pybind11::arg("v"));
		cl.def_static("remainder", (class Tensor * (*)(class Tensor *, float)) &Tensor::remainder, "C++: Tensor::remainder(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def("round_", (void (Tensor::*)()) &Tensor::round_, "C++: Tensor::round_() --> void");
		cl.def_static("round", (class Tensor * (*)(class Tensor *)) &Tensor::round, "C++: Tensor::round(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("rsqrt_", (void (Tensor::*)()) &Tensor::rsqrt_, "C++: Tensor::rsqrt_() --> void");
		cl.def_static("rsqrt", (class Tensor * (*)(class Tensor *)) &Tensor::rsqrt, "C++: Tensor::rsqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sigmoid_", (void (Tensor::*)()) &Tensor::sigmoid_, "C++: Tensor::sigmoid_() --> void");
		cl.def_static("sigmoid", (class Tensor * (*)(class Tensor *)) &Tensor::sigmoid, "C++: Tensor::sigmoid(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sign_", (void (Tensor::*)()) &Tensor::sign_, "C++: Tensor::sign_() --> void");
		cl.def_static("sign", (class Tensor * (*)(class Tensor *)) &Tensor::sign, "C++: Tensor::sign(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("sign", (void (*)(class Tensor *, class Tensor *)) &Tensor::sign, "C++: Tensor::sign(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sin_", (void (Tensor::*)()) &Tensor::sin_, "C++: Tensor::sin_() --> void");
		cl.def_static("sin", (class Tensor * (*)(class Tensor *)) &Tensor::sin, "C++: Tensor::sin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sinh_", (void (Tensor::*)()) &Tensor::sinh_, "C++: Tensor::sinh_() --> void");
		cl.def_static("sinh", (class Tensor * (*)(class Tensor *)) &Tensor::sinh, "C++: Tensor::sinh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sqr_", (void (Tensor::*)()) &Tensor::sqr_, "C++: Tensor::sqr_() --> void");
		cl.def_static("sqr", (class Tensor * (*)(class Tensor *)) &Tensor::sqr, "C++: Tensor::sqr(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sqrt_", (void (Tensor::*)()) &Tensor::sqrt_, "C++: Tensor::sqrt_() --> void");
		cl.def_static("sqrt", (class Tensor * (*)(class Tensor *)) &Tensor::sqrt, "C++: Tensor::sqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sub_", (void (Tensor::*)(float)) &Tensor::sub_, "C++: Tensor::sub_(float) --> void", pybind11::arg("v"));
		cl.def_static("sub", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::sub, "C++: Tensor::sub(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sum", (float (Tensor::*)()) &Tensor::sum, "C++: Tensor::sum() --> float");
		cl.def_static("sum2D_rowwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_rowwise, "C++: Tensor::sum2D_rowwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("sum2D_colwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_colwise, "C++: Tensor::sum2D_colwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("sum_abs", (float (Tensor::*)()) &Tensor::sum_abs, "C++: Tensor::sum_abs() --> float");
		cl.def_static("static_sum_abs", (class Tensor * (*)(class Tensor *)) &Tensor::sum_abs, "C++: Tensor::sum_abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("tan_", (void (Tensor::*)()) &Tensor::tan_, "C++: Tensor::tan_() --> void");
		cl.def_static("tan", (class Tensor * (*)(class Tensor *)) &Tensor::tan, "C++: Tensor::tan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("tanh_", (void (Tensor::*)()) &Tensor::tanh_, "C++: Tensor::tanh_() --> void");
		cl.def_static("tanh", (class Tensor * (*)(class Tensor *)) &Tensor::tanh, "C++: Tensor::tanh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("trunc_", (void (Tensor::*)()) &Tensor::trunc_, "C++: Tensor::trunc_() --> void");
		cl.def_static("trunc", (class Tensor * (*)(class Tensor *)) &Tensor::trunc, "C++: Tensor::trunc(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("reduce_sum2D", (void (*)(class Tensor *, class Tensor *, int, int)) &Tensor::reduce_sum2D, "C++: Tensor::reduce_sum2D(class Tensor *, class Tensor *, int, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"), pybind11::arg("incB"));
		cl.def_static("reduceTosum", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::reduceTosum, "C++: Tensor::reduceTosum(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def_static("eqsize", (int (*)(class Tensor *, class Tensor *)) &Tensor::eqsize, "C++: Tensor::eqsize(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal", [](class Tensor * a0, class Tensor * a1) -> int { return Tensor::equal(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal", (int (*)(class Tensor *, class Tensor *, float)) &Tensor::equal, "C++: Tensor::equal(class Tensor *, class Tensor *, float) --> int", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("epsilon"));
		cl.def_static("copy", (void (*)(class Tensor *, class Tensor *)) &Tensor::copy, "C++: Tensor::copy(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("fill", (void (*)(class Tensor *, int, int, class Tensor *, int, int, int)) &Tensor::fill, "C++: Tensor::fill(class Tensor *, int, int, class Tensor *, int, int, int) --> void", pybind11::arg("A"), pybind11::arg("aini"), pybind11::arg("aend"), pybind11::arg("B"), pybind11::arg("bini"), pybind11::arg("bend"), pybind11::arg("inc"));
		cl.def("rand_uniform", (void (Tensor::*)(float)) &Tensor::rand_uniform, "C++: Tensor::rand_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_signed_uniform", (void (Tensor::*)(float)) &Tensor::rand_signed_uniform, "C++: Tensor::rand_signed_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_normal", [](Tensor &o, float const & a0, float const & a1) -> void { return o.rand_normal(a0, a1); }, "", pybind11::arg("m"), pybind11::arg("s"));
		cl.def("rand_normal", (void (Tensor::*)(float, float, bool)) &Tensor::rand_normal, "C++: Tensor::rand_normal(float, float, bool) --> void", pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("rand_binary", (void (Tensor::*)(float)) &Tensor::rand_binary, "C++: Tensor::rand_binary(float) --> void", pybind11::arg("v"));

		tensor_addons(cl);
	}
}


// File: eddl/descriptors/descriptors.cpp
#include <compserv_addons.hpp>
#include <convoldescriptor_addons.hpp>
#include <eddl/compserv.h>
#include <eddl/descriptors/descriptors.h>
#include <eddl/initializers/initializer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/tensor/nn/tensor_nn.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <memory>
#include <pooldescriptor_addons.hpp>
#include <reducedescriptor_addons.hpp>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// Loss file:eddl/losses/loss.h line:22
struct PyCallBack_Loss : public Loss {
	using Loss::Loss;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Loss *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Loss::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Loss *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return Loss::value(a0, a1);
	}
};

// LMeanSquaredError file:eddl/losses/loss.h line:33
struct PyCallBack_LMeanSquaredError : public LMeanSquaredError {
	using LMeanSquaredError::LMeanSquaredError;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMeanSquaredError *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMeanSquaredError::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMeanSquaredError *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LMeanSquaredError::value(a0, a1);
	}
};

// LCrossEntropy file:eddl/losses/loss.h line:42
struct PyCallBack_LCrossEntropy : public LCrossEntropy {
	using LCrossEntropy::LCrossEntropy;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LCrossEntropy *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LCrossEntropy::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LCrossEntropy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LCrossEntropy::value(a0, a1);
	}
};

// LSoftCrossEntropy file:eddl/losses/loss.h line:51
struct PyCallBack_LSoftCrossEntropy : public LSoftCrossEntropy {
	using LSoftCrossEntropy::LSoftCrossEntropy;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSoftCrossEntropy *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LSoftCrossEntropy::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSoftCrossEntropy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LSoftCrossEntropy::value(a0, a1);
	}
};

// Metric file:eddl/metrics/metric.h line:23
struct PyCallBack_Metric : public Metric {
	using Metric::Metric;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Metric *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return Metric::value(a0, a1);
	}
};

// MMeanSquaredError file:eddl/metrics/metric.h line:33
struct PyCallBack_MMeanSquaredError : public MMeanSquaredError {
	using MMeanSquaredError::MMeanSquaredError;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MMeanSquaredError *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return MMeanSquaredError::value(a0, a1);
	}
};

// MCategoricalAccuracy file:eddl/metrics/metric.h line:41
struct PyCallBack_MCategoricalAccuracy : public MCategoricalAccuracy {
	using MCategoricalAccuracy::MCategoricalAccuracy;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MCategoricalAccuracy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return MCategoricalAccuracy::value(a0, a1);
	}
};

// Initializer file:eddl/initializers/initializer.h line:21
struct PyCallBack_Initializer : public Initializer {
	using Initializer::Initializer;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Initializer *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Initializer::apply\"");
	}
};

// IConstant file:eddl/initializers/initializer.h line:29
struct PyCallBack_IConstant : public IConstant {
	using IConstant::IConstant;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IConstant *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IConstant::apply(a0);
	}
};

// IIdentity file:eddl/initializers/initializer.h line:37
struct PyCallBack_IIdentity : public IIdentity {
	using IIdentity::IIdentity;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IIdentity *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IIdentity::apply(a0);
	}
};

// IGlorotNormal file:eddl/initializers/initializer.h line:45
struct PyCallBack_IGlorotNormal : public IGlorotNormal {
	using IGlorotNormal::IGlorotNormal;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IGlorotNormal *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IGlorotNormal::apply(a0);
	}
};

void bind_eddl_descriptors_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // ReduceDescriptor file:eddl/descriptors/descriptors.h line:25
		pybind11::class_<ReduceDescriptor, std::shared_ptr<ReduceDescriptor>> cl(M(""), "ReduceDescriptor", "");
		cl.def( pybind11::init( [](){ return new ReduceDescriptor(); } ) );
		cl.def( pybind11::init( [](ReduceDescriptor const &o){ return new ReduceDescriptor(o); } ) );
		cl.def_readwrite("axis", &ReduceDescriptor::axis);
		cl.def_readwrite("keepdims", &ReduceDescriptor::keepdims);
		cl.def_readwrite("m", &ReduceDescriptor::m);
		cl.def_readwrite("red_size", &ReduceDescriptor::red_size);
		cl.def_readwrite("index", &ReduceDescriptor::index);
		cl.def_readwrite("factor", &ReduceDescriptor::factor);
		cl.def("resize", (void (ReduceDescriptor::*)(int)) &ReduceDescriptor::resize, "C++: ReduceDescriptor::resize(int) --> void", pybind11::arg("b"));
		cl.def("build_index", (void (ReduceDescriptor::*)()) &ReduceDescriptor::build_index, "C++: ReduceDescriptor::build_index() --> void");

		reducedescriptor_addons(cl);
	}
	{ // ConvolDescriptor file:eddl/descriptors/descriptors.h line:50
		pybind11::class_<ConvolDescriptor, std::shared_ptr<ConvolDescriptor>> cl(M(""), "ConvolDescriptor", "");
		cl.def( pybind11::init( [](){ return new ConvolDescriptor(); } ) );
		cl.def( pybind11::init( [](ConvolDescriptor const &o){ return new ConvolDescriptor(o); } ) );
		cl.def_readwrite("ksize", &ConvolDescriptor::ksize);
		cl.def_readwrite("stride", &ConvolDescriptor::stride);
		cl.def_readwrite("pad", &ConvolDescriptor::pad);
		cl.def_readwrite("nk", &ConvolDescriptor::nk);
		cl.def_readwrite("kr", &ConvolDescriptor::kr);
		cl.def_readwrite("kc", &ConvolDescriptor::kc);
		cl.def_readwrite("kz", &ConvolDescriptor::kz);
		cl.def_readwrite("sr", &ConvolDescriptor::sr);
		cl.def_readwrite("sc", &ConvolDescriptor::sc);
		cl.def_readwrite("ir", &ConvolDescriptor::ir);
		cl.def_readwrite("ic", &ConvolDescriptor::ic);
		cl.def_readwrite("iz", &ConvolDescriptor::iz);
		cl.def_readwrite("r", &ConvolDescriptor::r);
		cl.def_readwrite("c", &ConvolDescriptor::c);
		cl.def_readwrite("z", &ConvolDescriptor::z);
		cl.def_readwrite("padr", &ConvolDescriptor::padr);
		cl.def_readwrite("padc", &ConvolDescriptor::padc);
		cl.def_readwrite("matI", &ConvolDescriptor::matI);
		cl.def_readwrite("matK", &ConvolDescriptor::matK);
		cl.def_readwrite("matO", &ConvolDescriptor::matO);
		cl.def_readwrite("matD", &ConvolDescriptor::matD);
		cl.def_readwrite("matgK", &ConvolDescriptor::matgK);
		cl.def("build", (void (ConvolDescriptor::*)(class Tensor *)) &ConvolDescriptor::build, "C++: ConvolDescriptor::build(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("resize", (void (ConvolDescriptor::*)(int)) &ConvolDescriptor::resize, "C++: ConvolDescriptor::resize(int) --> void", pybind11::arg("b"));

		convoldescriptor_addons(cl);
	}
	{ // PoolDescriptor file:eddl/descriptors/descriptors.h line:100
		pybind11::class_<PoolDescriptor, std::shared_ptr<PoolDescriptor>, ConvolDescriptor> cl(M(""), "PoolDescriptor", "");
		cl.def("build", (void (PoolDescriptor::*)(class Tensor *)) &PoolDescriptor::build, "C++: PoolDescriptor::build(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("resize", (void (PoolDescriptor::*)(int)) &PoolDescriptor::resize, "C++: PoolDescriptor::resize(int) --> void", pybind11::arg("b"));

		pooldescriptor_addons(cl);
	}
	// cent(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:20
	M("").def("cent", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &cent, "C++: cent(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));

	// accuracy(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:23
	M("").def("accuracy", (int (*)(class Tensor *, class Tensor *)) &accuracy, "C++: accuracy(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));

	// ReLu(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:28
	M("").def("ReLu", (void (*)(class Tensor *, class Tensor *)) &ReLu, "C++: ReLu(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// D_ReLu(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:29
	M("").def("D_ReLu", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &D_ReLu, "C++: D_ReLu(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("I"), pybind11::arg("PD"));

	// Sigmoid(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:32
	M("").def("Sigmoid", (void (*)(class Tensor *, class Tensor *)) &Sigmoid, "C++: Sigmoid(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// D_Sigmoid(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:33
	M("").def("D_Sigmoid", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &D_Sigmoid, "C++: D_Sigmoid(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("I"), pybind11::arg("PD"));

	// Softmax(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:36
	M("").def("Softmax", (void (*)(class Tensor *, class Tensor *)) &Softmax, "C++: Softmax(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// D_Softmax(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:37
	M("").def("D_Softmax", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &D_Softmax, "C++: D_Softmax(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("I"), pybind11::arg("PD"));

	// Conv2D(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:42
	M("").def("Conv2D", (void (*)(class ConvolDescriptor *)) &Conv2D, "C++: Conv2D(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// Conv2D_grad(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:43
	M("").def("Conv2D_grad", (void (*)(class ConvolDescriptor *)) &Conv2D_grad, "C++: Conv2D_grad(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// Conv2D_back(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:44
	M("").def("Conv2D_back", (void (*)(class ConvolDescriptor *)) &Conv2D_back, "C++: Conv2D_back(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// MPool2D(class PoolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:47
	M("").def("MPool2D", (void (*)(class PoolDescriptor *)) &MPool2D, "C++: MPool2D(class PoolDescriptor *) --> void", pybind11::arg("D"));

	// MPool2D_back(class PoolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:48
	M("").def("MPool2D_back", (void (*)(class PoolDescriptor *)) &MPool2D_back, "C++: MPool2D_back(class PoolDescriptor *) --> void", pybind11::arg("D"));

	{ // CompServ file:eddl/compserv.h line:21
		pybind11::class_<CompServ, std::shared_ptr<CompServ>> cl(M(""), "CompServ", "");
		cl.def( pybind11::init<struct _IO_FILE *>(), pybind11::arg("csspec") );

		cl.def( pybind11::init( [](CompServ const &o){ return new CompServ(o); } ) );
		cl.def_readwrite("type", &CompServ::type);
		cl.def_readwrite("local_threads", &CompServ::local_threads);
		cl.def_readwrite("local_gpus", &CompServ::local_gpus);
		cl.def_readwrite("local_fpgas", &CompServ::local_fpgas);
		cl.def_readwrite("lsb", &CompServ::lsb);

		compserv_addons(cl);
	}
	{ // Loss file:eddl/losses/loss.h line:22
		pybind11::class_<Loss, std::shared_ptr<Loss>, PyCallBack_Loss> cl(M(""), "Loss", "");
		cl.def( pybind11::init( [](PyCallBack_Loss const &o){ return new PyCallBack_Loss(o); } ) );
		cl.def( pybind11::init( [](Loss const &o){ return new Loss(o); } ) );
		cl.def_readwrite("name", &Loss::name);
		cl.def("delta", (void (Loss::*)(class Tensor *, class Tensor *, class Tensor *)) &Loss::delta, "C++: Loss::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (Loss::*)(class Tensor *, class Tensor *)) &Loss::value, "C++: Loss::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Loss & (Loss::*)(const class Loss &)) &Loss::operator=, "C++: Loss::operator=(const class Loss &) --> class Loss &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMeanSquaredError file:eddl/losses/loss.h line:33
		pybind11::class_<LMeanSquaredError, std::shared_ptr<LMeanSquaredError>, PyCallBack_LMeanSquaredError, Loss> cl(M(""), "LMeanSquaredError", "");
		cl.def( pybind11::init( [](){ return new LMeanSquaredError(); }, [](){ return new PyCallBack_LMeanSquaredError(); } ) );
		cl.def("delta", (void (LMeanSquaredError::*)(class Tensor *, class Tensor *, class Tensor *)) &LMeanSquaredError::delta, "C++: LMeanSquaredError::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LMeanSquaredError::*)(class Tensor *, class Tensor *)) &LMeanSquaredError::value, "C++: LMeanSquaredError::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LMeanSquaredError & (LMeanSquaredError::*)(const class LMeanSquaredError &)) &LMeanSquaredError::operator=, "C++: LMeanSquaredError::operator=(const class LMeanSquaredError &) --> class LMeanSquaredError &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LCrossEntropy file:eddl/losses/loss.h line:42
		pybind11::class_<LCrossEntropy, std::shared_ptr<LCrossEntropy>, PyCallBack_LCrossEntropy, Loss> cl(M(""), "LCrossEntropy", "");
		cl.def( pybind11::init( [](){ return new LCrossEntropy(); }, [](){ return new PyCallBack_LCrossEntropy(); } ) );
		cl.def("delta", (void (LCrossEntropy::*)(class Tensor *, class Tensor *, class Tensor *)) &LCrossEntropy::delta, "C++: LCrossEntropy::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LCrossEntropy::*)(class Tensor *, class Tensor *)) &LCrossEntropy::value, "C++: LCrossEntropy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LCrossEntropy & (LCrossEntropy::*)(const class LCrossEntropy &)) &LCrossEntropy::operator=, "C++: LCrossEntropy::operator=(const class LCrossEntropy &) --> class LCrossEntropy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LSoftCrossEntropy file:eddl/losses/loss.h line:51
		pybind11::class_<LSoftCrossEntropy, std::shared_ptr<LSoftCrossEntropy>, PyCallBack_LSoftCrossEntropy, Loss> cl(M(""), "LSoftCrossEntropy", "");
		cl.def( pybind11::init( [](){ return new LSoftCrossEntropy(); }, [](){ return new PyCallBack_LSoftCrossEntropy(); } ) );
		cl.def("delta", (void (LSoftCrossEntropy::*)(class Tensor *, class Tensor *, class Tensor *)) &LSoftCrossEntropy::delta, "C++: LSoftCrossEntropy::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LSoftCrossEntropy::*)(class Tensor *, class Tensor *)) &LSoftCrossEntropy::value, "C++: LSoftCrossEntropy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LSoftCrossEntropy & (LSoftCrossEntropy::*)(const class LSoftCrossEntropy &)) &LSoftCrossEntropy::operator=, "C++: LSoftCrossEntropy::operator=(const class LSoftCrossEntropy &) --> class LSoftCrossEntropy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Metric file:eddl/metrics/metric.h line:23
		pybind11::class_<Metric, std::shared_ptr<Metric>, PyCallBack_Metric> cl(M(""), "Metric", "");
		cl.def( pybind11::init( [](PyCallBack_Metric const &o){ return new PyCallBack_Metric(o); } ) );
		cl.def( pybind11::init( [](Metric const &o){ return new Metric(o); } ) );
		cl.def_readwrite("name", &Metric::name);
		cl.def("value", (float (Metric::*)(class Tensor *, class Tensor *)) &Metric::value, "C++: Metric::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Metric & (Metric::*)(const class Metric &)) &Metric::operator=, "C++: Metric::operator=(const class Metric &) --> class Metric &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // MMeanSquaredError file:eddl/metrics/metric.h line:33
		pybind11::class_<MMeanSquaredError, std::shared_ptr<MMeanSquaredError>, PyCallBack_MMeanSquaredError, Metric> cl(M(""), "MMeanSquaredError", "");
		cl.def( pybind11::init( [](){ return new MMeanSquaredError(); }, [](){ return new PyCallBack_MMeanSquaredError(); } ) );
		cl.def("value", (float (MMeanSquaredError::*)(class Tensor *, class Tensor *)) &MMeanSquaredError::value, "C++: MMeanSquaredError::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class MMeanSquaredError & (MMeanSquaredError::*)(const class MMeanSquaredError &)) &MMeanSquaredError::operator=, "C++: MMeanSquaredError::operator=(const class MMeanSquaredError &) --> class MMeanSquaredError &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // MCategoricalAccuracy file:eddl/metrics/metric.h line:41
		pybind11::class_<MCategoricalAccuracy, std::shared_ptr<MCategoricalAccuracy>, PyCallBack_MCategoricalAccuracy, Metric> cl(M(""), "MCategoricalAccuracy", "");
		cl.def( pybind11::init( [](){ return new MCategoricalAccuracy(); }, [](){ return new PyCallBack_MCategoricalAccuracy(); } ) );
		cl.def("value", (float (MCategoricalAccuracy::*)(class Tensor *, class Tensor *)) &MCategoricalAccuracy::value, "C++: MCategoricalAccuracy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class MCategoricalAccuracy & (MCategoricalAccuracy::*)(const class MCategoricalAccuracy &)) &MCategoricalAccuracy::operator=, "C++: MCategoricalAccuracy::operator=(const class MCategoricalAccuracy &) --> class MCategoricalAccuracy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Initializer file:eddl/initializers/initializer.h line:21
		pybind11::class_<Initializer, std::shared_ptr<Initializer>, PyCallBack_Initializer> cl(M(""), "Initializer", "");
		cl.def(pybind11::init<PyCallBack_Initializer const &>());
		cl.def_readwrite("name", &Initializer::name);
		cl.def("apply", (void (Initializer::*)(class Tensor *)) &Initializer::apply, "C++: Initializer::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class Initializer & (Initializer::*)(const class Initializer &)) &Initializer::operator=, "C++: Initializer::operator=(const class Initializer &) --> class Initializer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IConstant file:eddl/initializers/initializer.h line:29
		pybind11::class_<IConstant, std::shared_ptr<IConstant>, PyCallBack_IConstant, Initializer> cl(M(""), "IConstant", "");
		cl.def( pybind11::init<float>(), pybind11::arg("value") );

		cl.def_readwrite("value", &IConstant::value);
		cl.def("apply", (void (IConstant::*)(class Tensor *)) &IConstant::apply, "C++: IConstant::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IConstant & (IConstant::*)(const class IConstant &)) &IConstant::operator=, "C++: IConstant::operator=(const class IConstant &) --> class IConstant &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IIdentity file:eddl/initializers/initializer.h line:37
		pybind11::class_<IIdentity, std::shared_ptr<IIdentity>, PyCallBack_IIdentity, Initializer> cl(M(""), "IIdentity", "");
		cl.def( pybind11::init<float>(), pybind11::arg("gain") );

		cl.def_readwrite("gain", &IIdentity::gain);
		cl.def("apply", (void (IIdentity::*)(class Tensor *)) &IIdentity::apply, "C++: IIdentity::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IIdentity & (IIdentity::*)(const class IIdentity &)) &IIdentity::operator=, "C++: IIdentity::operator=(const class IIdentity &) --> class IIdentity &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IGlorotNormal file:eddl/initializers/initializer.h line:45
		pybind11::class_<IGlorotNormal, std::shared_ptr<IGlorotNormal>, PyCallBack_IGlorotNormal, Initializer> cl(M(""), "IGlorotNormal", "");
		cl.def( pybind11::init( [](){ return new IGlorotNormal(); }, [](){ return new PyCallBack_IGlorotNormal(); } ), "doc");
		cl.def( pybind11::init<int>(), pybind11::arg("seed") );

		cl.def_readwrite("seed", &IGlorotNormal::seed);
		cl.def("apply", (void (IGlorotNormal::*)(class Tensor *)) &IGlorotNormal::apply, "C++: IGlorotNormal::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IGlorotNormal & (IGlorotNormal::*)(const class IGlorotNormal &)) &IGlorotNormal::operator=, "C++: IGlorotNormal::operator=(const class IGlorotNormal &) --> class IGlorotNormal &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/initializers/initializer.cpp
#include <eddl/descriptors/descriptors.h>
#include <eddl/initializers/initializer.h>
#include <eddl/layers/core/layer_core.h>
#include <eddl/layers/layer.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <lactivation_addons.hpp>
#include <layer_addons.hpp>
#include <ldense_addons.hpp>
#include <lembedding_addons.hpp>
#include <linput_addons.hpp>
#include <ltensor_addons.hpp>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// IGlorotUniform file:eddl/initializers/initializer.h line:53
struct PyCallBack_IGlorotUniform : public IGlorotUniform {
	using IGlorotUniform::IGlorotUniform;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IGlorotUniform *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IGlorotUniform::apply(a0);
	}
};

// IRandomNormal file:eddl/initializers/initializer.h line:61
struct PyCallBack_IRandomNormal : public IRandomNormal {
	using IRandomNormal::IRandomNormal;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IRandomNormal *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IRandomNormal::apply(a0);
	}
};

// IRandomUniform file:eddl/initializers/initializer.h line:71
struct PyCallBack_IRandomUniform : public IRandomUniform {
	using IRandomUniform::IRandomUniform;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IRandomUniform *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IRandomUniform::apply(a0);
	}
};

// IOrthogonal file:eddl/initializers/initializer.h line:81
struct PyCallBack_IOrthogonal : public IOrthogonal {
	using IOrthogonal::IOrthogonal;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const IOrthogonal *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOrthogonal::apply(a0);
	}
};

// Layer file:eddl/layers/layer.h line:29
struct PyCallBack_Layer : public Layer {
	using Layer::Layer;

	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::backward();
	}
};

// LinLayer file:eddl/layers/layer.h line:107
struct PyCallBack_LinLayer : public LinLayer {
	using LinLayer::LinLayer;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// MLayer file:eddl/layers/layer.h line:135
struct PyCallBack_MLayer : public MLayer {
	using MLayer::MLayer;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// Regularizer file: line:21
struct PyCallBack_Regularizer : public Regularizer {
	using Regularizer::Regularizer;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Regularizer *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Regularizer::apply\"");
	}
};

// RL1 file: line:29
struct PyCallBack_RL1 : public RL1 {
	using RL1::RL1;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RL1 *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return RL1::apply(a0);
	}
};

// RL2 file: line:37
struct PyCallBack_RL2 : public RL2 {
	using RL2::RL2;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RL2 *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return RL2::apply(a0);
	}
};

// RL1L2 file: line:45
struct PyCallBack_RL1L2 : public RL1L2 {
	using RL1L2::RL1L2;

	void apply(class Tensor * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RL1L2 *>(this), "apply");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return RL1L2::apply(a0);
	}
};

// LTensor file:eddl/layers/core/layer_core.h line:26
struct PyCallBack_LTensor : public LTensor {
	using LTensor::LTensor;

	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::info();
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LInput file:eddl/layers/core/layer_core.h line:62
struct PyCallBack_LInput : public LInput {
	using LInput::LInput;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LEmbedding file:eddl/layers/core/layer_core.h line:84
struct PyCallBack_LEmbedding : public LEmbedding {
	using LEmbedding::LEmbedding;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LDense file:eddl/layers/core/layer_core.h line:107
struct PyCallBack_LDense : public LDense {
	using LDense::LDense;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LActivation file:eddl/layers/core/layer_core.h line:137
struct PyCallBack_LActivation : public LActivation {
	using LActivation::LActivation;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

void bind_eddl_initializers_initializer(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // IGlorotUniform file:eddl/initializers/initializer.h line:53
		pybind11::class_<IGlorotUniform, std::shared_ptr<IGlorotUniform>, PyCallBack_IGlorotUniform, Initializer> cl(M(""), "IGlorotUniform", "");
		cl.def( pybind11::init( [](){ return new IGlorotUniform(); }, [](){ return new PyCallBack_IGlorotUniform(); } ), "doc");
		cl.def( pybind11::init<int>(), pybind11::arg("seed") );

		cl.def_readwrite("seed", &IGlorotUniform::seed);
		cl.def("apply", (void (IGlorotUniform::*)(class Tensor *)) &IGlorotUniform::apply, "C++: IGlorotUniform::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IGlorotUniform & (IGlorotUniform::*)(const class IGlorotUniform &)) &IGlorotUniform::operator=, "C++: IGlorotUniform::operator=(const class IGlorotUniform &) --> class IGlorotUniform &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IRandomNormal file:eddl/initializers/initializer.h line:61
		pybind11::class_<IRandomNormal, std::shared_ptr<IRandomNormal>, PyCallBack_IRandomNormal, Initializer> cl(M(""), "IRandomNormal", "");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new IRandomNormal(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_IRandomNormal(a0, a1); } ), "doc");
		cl.def( pybind11::init<float, float, int>(), pybind11::arg("mean"), pybind11::arg("stdev"), pybind11::arg("seed") );

		cl.def_readwrite("mean", &IRandomNormal::mean);
		cl.def_readwrite("stdev", &IRandomNormal::stdev);
		cl.def_readwrite("seed", &IRandomNormal::seed);
		cl.def("apply", (void (IRandomNormal::*)(class Tensor *)) &IRandomNormal::apply, "C++: IRandomNormal::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IRandomNormal & (IRandomNormal::*)(const class IRandomNormal &)) &IRandomNormal::operator=, "C++: IRandomNormal::operator=(const class IRandomNormal &) --> class IRandomNormal &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IRandomUniform file:eddl/initializers/initializer.h line:71
		pybind11::class_<IRandomUniform, std::shared_ptr<IRandomUniform>, PyCallBack_IRandomUniform, Initializer> cl(M(""), "IRandomUniform", "");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new IRandomUniform(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_IRandomUniform(a0, a1); } ), "doc");
		cl.def( pybind11::init<float, float, int>(), pybind11::arg("minval"), pybind11::arg("maxval"), pybind11::arg("seed") );

		cl.def_readwrite("minval", &IRandomUniform::minval);
		cl.def_readwrite("maxval", &IRandomUniform::maxval);
		cl.def_readwrite("seed", &IRandomUniform::seed);
		cl.def("apply", (void (IRandomUniform::*)(class Tensor *)) &IRandomUniform::apply, "C++: IRandomUniform::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IRandomUniform & (IRandomUniform::*)(const class IRandomUniform &)) &IRandomUniform::operator=, "C++: IRandomUniform::operator=(const class IRandomUniform &) --> class IRandomUniform &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // IOrthogonal file:eddl/initializers/initializer.h line:81
		pybind11::class_<IOrthogonal, std::shared_ptr<IOrthogonal>, PyCallBack_IOrthogonal, Initializer> cl(M(""), "IOrthogonal", "");
		cl.def( pybind11::init( [](float const & a0){ return new IOrthogonal(a0); }, [](float const & a0){ return new PyCallBack_IOrthogonal(a0); } ), "doc");
		cl.def( pybind11::init<float, int>(), pybind11::arg("gain"), pybind11::arg("seed") );

		cl.def_readwrite("gain", &IOrthogonal::gain);
		cl.def_readwrite("seed", &IOrthogonal::seed);
		cl.def("apply", (void (IOrthogonal::*)(class Tensor *)) &IOrthogonal::apply, "C++: IOrthogonal::apply(class Tensor *) --> void", pybind11::arg("params"));
		cl.def("assign", (class IOrthogonal & (IOrthogonal::*)(const class IOrthogonal &)) &IOrthogonal::operator=, "C++: IOrthogonal::operator=(const class IOrthogonal &) --> class IOrthogonal &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// reduction(class ReduceDescriptor *) file: line:19
	M("").def("reduction", (void (*)(class ReduceDescriptor *)) &reduction, "C++: reduction(class ReduceDescriptor *) --> void", pybind11::arg("RD"));

	// reduction_back(class ReduceDescriptor *) file: line:20
	M("").def("reduction_back", (void (*)(class ReduceDescriptor *)) &reduction_back, "C++: reduction_back(class ReduceDescriptor *) --> void", pybind11::arg("RD"));

	{ // Layer file:eddl/layers/layer.h line:29
		pybind11::class_<Layer, std::shared_ptr<Layer>, PyCallBack_Layer> cl(M(""), "Layer", "");
		cl.def( pybind11::init( [](PyCallBack_Layer const &o){ return new PyCallBack_Layer(o); } ) );
		cl.def( pybind11::init( [](Layer const &o){ return new Layer(o); } ) );
		cl.def_readwrite("name", &Layer::name);
		cl.def_readwrite("params", &Layer::params);
		cl.def_readwrite("gradients", &Layer::gradients);
		cl.def_readwrite("parent", &Layer::parent);
		cl.def_readwrite("child", &Layer::child);
		cl.def_readwrite("mode", &Layer::mode);
		cl.def_readwrite("dev", &Layer::dev);
		cl.def_readwrite("lin", &Layer::lin);
		cl.def_readwrite("lout", &Layer::lout);
		cl.def_readwrite("delta_bp", &Layer::delta_bp);
		cl.def("initialize", (void (Layer::*)(class Initializer *)) &Layer::initialize, "C++: Layer::initialize(class Initializer *) --> void", pybind11::arg("init"));
		cl.def("info", (void (Layer::*)()) &Layer::info, "C++: Layer::info() --> void");
		cl.def("setmode", (void (Layer::*)(int)) &Layer::setmode, "C++: Layer::setmode(int) --> void", pybind11::arg("m"));
		cl.def("detach", (void (Layer::*)(class Layer *)) &Layer::detach, "C++: Layer::detach(class Layer *) --> void", pybind11::arg("l"));
		cl.def("getWeights", (class Tensor * (Layer::*)()) &Layer::getWeights, "C++: Layer::getWeights() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setWeights", (class Tensor * (Layer::*)(class Tensor)) &Layer::setWeights, "C++: Layer::setWeights(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("getBias", (class Tensor * (Layer::*)()) &Layer::getBias, "C++: Layer::getBias() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setBias", (class Tensor * (Layer::*)(class Tensor)) &Layer::setBias, "C++: Layer::setBias(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("reset", (void (Layer::*)()) &Layer::reset, "C++: Layer::reset() --> void");
		cl.def("resize", (void (Layer::*)(int)) &Layer::resize, "C++: Layer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("addchild", (void (Layer::*)(class Layer *)) &Layer::addchild, "C++: Layer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (Layer::*)(class Layer *)) &Layer::addparent, "C++: Layer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("forward", (void (Layer::*)()) &Layer::forward, "C++: Layer::forward() --> void");
		cl.def("backward", (void (Layer::*)()) &Layer::backward, "C++: Layer::backward() --> void");
		cl.def("assign", (class Layer & (Layer::*)(const class Layer &)) &Layer::operator=, "C++: Layer::operator=(const class Layer &) --> class Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		layer_addons(cl);
	}
	{ // LinLayer file:eddl/layers/layer.h line:107
		pybind11::class_<LinLayer, std::shared_ptr<LinLayer>, PyCallBack_LinLayer, Layer> cl(M(""), "LinLayer", "//////////////////////////////////////\n//////////////////////////////////////");
		cl.def( pybind11::init( [](PyCallBack_LinLayer const &o){ return new PyCallBack_LinLayer(o); } ) );
		cl.def( pybind11::init( [](LinLayer const &o){ return new LinLayer(o); } ) );
		cl.def("addchild", (void (LinLayer::*)(class Layer *)) &LinLayer::addchild, "C++: LinLayer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (LinLayer::*)(class Layer *)) &LinLayer::addparent, "C++: LinLayer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("resize", (void (LinLayer::*)(int)) &LinLayer::resize, "C++: LinLayer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (LinLayer::*)()) &LinLayer::forward, "C++: LinLayer::forward() --> void");
		cl.def("backward", (void (LinLayer::*)()) &LinLayer::backward, "C++: LinLayer::backward() --> void");
		cl.def("assign", (class LinLayer & (LinLayer::*)(const class LinLayer &)) &LinLayer::operator=, "C++: LinLayer::operator=(const class LinLayer &) --> class LinLayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // MLayer file:eddl/layers/layer.h line:135
		pybind11::class_<MLayer, std::shared_ptr<MLayer>, PyCallBack_MLayer, Layer> cl(M(""), "MLayer", "//////////////////////////////////////\n//////////////////////////////////////");
		cl.def( pybind11::init( [](PyCallBack_MLayer const &o){ return new PyCallBack_MLayer(o); } ) );
		cl.def( pybind11::init( [](MLayer const &o){ return new MLayer(o); } ) );
		cl.def("addchild", (void (MLayer::*)(class Layer *)) &MLayer::addchild, "C++: MLayer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (MLayer::*)(class Layer *)) &MLayer::addparent, "C++: MLayer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("resize", (void (MLayer::*)(int)) &MLayer::resize, "C++: MLayer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (MLayer::*)()) &MLayer::forward, "C++: MLayer::forward() --> void");
		cl.def("backward", (void (MLayer::*)()) &MLayer::backward, "C++: MLayer::backward() --> void");
		cl.def("assign", (class MLayer & (MLayer::*)(const class MLayer &)) &MLayer::operator=, "C++: MLayer::operator=(const class MLayer &) --> class MLayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Regularizer file: line:21
		pybind11::class_<Regularizer, std::shared_ptr<Regularizer>, PyCallBack_Regularizer> cl(M(""), "Regularizer", "");
		cl.def(pybind11::init<PyCallBack_Regularizer const &>());
		cl.def_readwrite("name", &Regularizer::name);
		cl.def("apply", (void (Regularizer::*)(class Tensor *)) &Regularizer::apply, "C++: Regularizer::apply(class Tensor *) --> void", pybind11::arg("T"));
		cl.def("assign", (class Regularizer & (Regularizer::*)(const class Regularizer &)) &Regularizer::operator=, "C++: Regularizer::operator=(const class Regularizer &) --> class Regularizer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // RL1 file: line:29
		pybind11::class_<RL1, std::shared_ptr<RL1>, PyCallBack_RL1, Regularizer> cl(M(""), "RL1", "");
		cl.def( pybind11::init<float>(), pybind11::arg("l1") );

		cl.def_readwrite("l1", &RL1::l1);
		cl.def("apply", (void (RL1::*)(class Tensor *)) &RL1::apply, "C++: RL1::apply(class Tensor *) --> void", pybind11::arg("T"));
		cl.def("assign", (class RL1 & (RL1::*)(const class RL1 &)) &RL1::operator=, "C++: RL1::operator=(const class RL1 &) --> class RL1 &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // RL2 file: line:37
		pybind11::class_<RL2, std::shared_ptr<RL2>, PyCallBack_RL2, Regularizer> cl(M(""), "RL2", "");
		cl.def( pybind11::init<float>(), pybind11::arg("l2") );

		cl.def_readwrite("l2", &RL2::l2);
		cl.def("apply", (void (RL2::*)(class Tensor *)) &RL2::apply, "C++: RL2::apply(class Tensor *) --> void", pybind11::arg("T"));
		cl.def("assign", (class RL2 & (RL2::*)(const class RL2 &)) &RL2::operator=, "C++: RL2::operator=(const class RL2 &) --> class RL2 &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // RL1L2 file: line:45
		pybind11::class_<RL1L2, std::shared_ptr<RL1L2>, PyCallBack_RL1L2, Regularizer> cl(M(""), "RL1L2", "");
		cl.def( pybind11::init<float, float>(), pybind11::arg("l1"), pybind11::arg("l2") );

		cl.def_readwrite("l1", &RL1L2::l1);
		cl.def_readwrite("l2", &RL1L2::l2);
		cl.def("apply", (void (RL1L2::*)(class Tensor *)) &RL1L2::apply, "C++: RL1L2::apply(class Tensor *) --> void", pybind11::arg("T"));
		cl.def("assign", (class RL1L2 & (RL1L2::*)(const class RL1L2 &)) &RL1L2::operator=, "C++: RL1L2::operator=(const class RL1L2 &) --> class RL1L2 &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LTensor file:eddl/layers/core/layer_core.h line:26
		pybind11::class_<LTensor, std::shared_ptr<LTensor>, PyCallBack_LTensor, LinLayer> cl(M(""), "LTensor", "Tensor Layer");
		cl.def( pybind11::init<class Layer *>(), pybind11::arg("l") );

		cl.def("info", (void (LTensor::*)()) &LTensor::info, "C++: LTensor::info() --> void");
		cl.def("forward", (void (LTensor::*)()) &LTensor::forward, "C++: LTensor::forward() --> void");
		cl.def("backward", (void (LTensor::*)()) &LTensor::backward, "C++: LTensor::backward() --> void");
		cl.def("resize", (void (LTensor::*)(int)) &LTensor::resize, "C++: LTensor::resize(int) --> void", pybind11::arg("batch"));
		cl.def("__add__", (class LTensor (LTensor::*)(class LTensor)) &LTensor::operator+, "C++: LTensor::operator+(class LTensor) --> class LTensor", pybind11::arg("L"));
		cl.def("assign", (class LTensor & (LTensor::*)(const class LTensor &)) &LTensor::operator=, "C++: LTensor::operator=(const class LTensor &) --> class LTensor &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		ltensor_addons(cl);
	}
	{ // LInput file:eddl/layers/core/layer_core.h line:62
		pybind11::class_<LInput, std::shared_ptr<LInput>, PyCallBack_LInput, LinLayer> cl(M(""), "LInput", "INPUT Layer");
		cl.def("forward", (void (LInput::*)()) &LInput::forward, "C++: LInput::forward() --> void");
		cl.def("backward", (void (LInput::*)()) &LInput::backward, "C++: LInput::backward() --> void");
		cl.def("resize", (void (LInput::*)(int)) &LInput::resize, "C++: LInput::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LInput & (LInput::*)(const class LInput &)) &LInput::operator=, "C++: LInput::operator=(const class LInput &) --> class LInput &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		linput_addons(cl);
	}
	{ // LEmbedding file:eddl/layers/core/layer_core.h line:84
		pybind11::class_<LEmbedding, std::shared_ptr<LEmbedding>, PyCallBack_LEmbedding, LinLayer> cl(M(""), "LEmbedding", "EMBEDDING Layer");
		cl.def_readwrite("input_dim", &LEmbedding::input_dim);
		cl.def_readwrite("output_dim", &LEmbedding::output_dim);
		cl.def("forward", (void (LEmbedding::*)()) &LEmbedding::forward, "C++: LEmbedding::forward() --> void");
		cl.def("backward", (void (LEmbedding::*)()) &LEmbedding::backward, "C++: LEmbedding::backward() --> void");
		cl.def("resize", (void (LEmbedding::*)(int)) &LEmbedding::resize, "C++: LEmbedding::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LEmbedding & (LEmbedding::*)(const class LEmbedding &)) &LEmbedding::operator=, "C++: LEmbedding::operator=(const class LEmbedding &) --> class LEmbedding &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lembedding_addons(cl);
	}
	{ // LDense file:eddl/layers/core/layer_core.h line:107
		pybind11::class_<LDense, std::shared_ptr<LDense>, PyCallBack_LDense, LinLayer> cl(M(""), "LDense", "Dense Layer");
		cl.def_readwrite("ndim", &LDense::ndim);
		cl.def_readwrite("use_bias", &LDense::use_bias);
		cl.def("forward", (void (LDense::*)()) &LDense::forward, "C++: LDense::forward() --> void");
		cl.def("backward", (void (LDense::*)()) &LDense::backward, "C++: LDense::backward() --> void");
		cl.def("resize", (void (LDense::*)(int)) &LDense::resize, "C++: LDense::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LDense & (LDense::*)(const class LDense &)) &LDense::operator=, "C++: LDense::operator=(const class LDense &) --> class LDense &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		ldense_addons(cl);
	}
	{ // LActivation file:eddl/layers/core/layer_core.h line:137
		pybind11::class_<LActivation, std::shared_ptr<LActivation>, PyCallBack_LActivation, LinLayer> cl(M(""), "LActivation", "Activation Layer");
		cl.def( pybind11::init( [](PyCallBack_LActivation const &o){ return new PyCallBack_LActivation(o); } ) );
		cl.def( pybind11::init( [](LActivation const &o){ return new LActivation(o); } ) );
		cl.def_readwrite("act", &LActivation::act);
		cl.def("forward", (void (LActivation::*)()) &LActivation::forward, "C++: LActivation::forward() --> void");
		cl.def("backward", (void (LActivation::*)()) &LActivation::backward, "C++: LActivation::backward() --> void");
		cl.def("resize", (void (LActivation::*)(int)) &LActivation::resize, "C++: LActivation::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LActivation & (LActivation::*)(const class LActivation &)) &LActivation::operator=, "C++: LActivation::operator=(const class LActivation &) --> class LActivation &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lactivation_addons(cl);
	}
}


// File: eddl/layers/core/layer_core.cpp
#include <eddl/descriptors/descriptors.h>
#include <eddl/initializers/initializer.h>
#include <eddl/layers/conv/layer_conv.h>
#include <eddl/layers/core/layer_core.h>
#include <eddl/layers/layer.h>
#include <eddl/layers/merge/layer_merge.h>
#include <eddl/layers/noise/layer_noise.h>
#include <eddl/layers/pool/layer_pool.h>
#include <eddl/layers/reductions/layer_reductions.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <lbatchnorm_addons.hpp>
#include <lconcat_addons.hpp>
#include <lconv_addons.hpp>
#include <ldropout_addons.hpp>
#include <lgaussiannoise_addons.hpp>
#include <lmaxpool_addons.hpp>
#include <lreshape_addons.hpp>
#include <lupsampling_addons.hpp>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// LReshape file:eddl/layers/core/layer_core.h line:159
struct PyCallBack_LReshape : public LReshape {
	using LReshape::LReshape;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LTranspose file:eddl/layers/core/layer_core.h line:184
struct PyCallBack_LTranspose : public LTranspose {
	using LTranspose::LTranspose;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LDropout file:eddl/layers/core/layer_core.h line:209
struct PyCallBack_LDropout : public LDropout {
	using LDropout::LDropout;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LBatchNorm file:eddl/layers/core/layer_core.h line:235
struct PyCallBack_LBatchNorm : public LBatchNorm {
	using LBatchNorm::LBatchNorm;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::resize(a0);
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::reset();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
};

// LConv file:eddl/layers/conv/layer_conv.h line:27
struct PyCallBack_LConv : public LConv {
	using LConv::LConv;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LConvT file:eddl/layers/conv/layer_conv.h line:60
struct PyCallBack_LConvT : public LConvT {
	using LConvT::LConvT;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LUpSampling file:eddl/layers/conv/layer_conv.h line:87
struct PyCallBack_LUpSampling : public LUpSampling {
	using LUpSampling::LUpSampling;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LUpSampling::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LUpSampling::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LUpSampling::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LPool file:eddl/layers/pool/layer_pool.h line:27
struct PyCallBack_LPool : public LPool {
	using LPool::LPool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LMaxPool file:eddl/layers/pool/layer_pool.h line:39
struct PyCallBack_LMaxPool : public LMaxPool {
	using LMaxPool::LMaxPool;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LAveragePool file:eddl/layers/pool/layer_pool.h line:68
struct PyCallBack_LAveragePool : public LAveragePool {
	using LAveragePool::LAveragePool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LGlobalMaxPool file:eddl/layers/pool/layer_pool.h line:92
struct PyCallBack_LGlobalMaxPool : public LGlobalMaxPool {
	using LGlobalMaxPool::LGlobalMaxPool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LGlobalAveragePool file:eddl/layers/pool/layer_pool.h line:115
struct PyCallBack_LGlobalAveragePool : public LGlobalAveragePool {
	using LGlobalAveragePool::LGlobalAveragePool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LGaussianNoise file:eddl/layers/noise/layer_noise.h line:26
struct PyCallBack_LGaussianNoise : public LGaussianNoise {
	using LGaussianNoise::LGaussianNoise;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LAdd file:eddl/layers/merge/layer_merge.h line:26
struct PyCallBack_LAdd : public LAdd {
	using LAdd::LAdd;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAdd::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAdd::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAdd::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAdd *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LSubtract file:eddl/layers/merge/layer_merge.h line:48
struct PyCallBack_LSubtract : public LSubtract {
	using LSubtract::LSubtract;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LSubtract::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LSubtract::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSubtract *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LMatMul file:eddl/layers/merge/layer_merge.h line:68
struct PyCallBack_LMatMul : public LMatMul {
	using LMatMul::LMatMul;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMatMul::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMatMul::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMatMul *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LAverage file:eddl/layers/merge/layer_merge.h line:88
struct PyCallBack_LAverage : public LAverage {
	using LAverage::LAverage;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAverage::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAverage::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LAverage::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAverage *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LMaximum file:eddl/layers/merge/layer_merge.h line:110
struct PyCallBack_LMaximum : public LMaximum {
	using LMaximum::LMaximum;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaximum::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaximum::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaximum *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LMinimum file:eddl/layers/merge/layer_merge.h line:130
struct PyCallBack_LMinimum : public LMinimum {
	using LMinimum::LMinimum;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMinimum::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMinimum::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMinimum *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LConcat file:eddl/layers/merge/layer_merge.h line:150
struct PyCallBack_LConcat : public LConcat {
	using LConcat::LConcat;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConcat::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConcat::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConcat::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConcat *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// ReductionLayer file:eddl/layers/reductions/layer_reductions.h line:30
struct PyCallBack_ReductionLayer : public ReductionLayer {
	using ReductionLayer::ReductionLayer;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReductionLayer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::backward();
	}
};

void bind_eddl_layers_core_layer_core(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // LReshape file:eddl/layers/core/layer_core.h line:159
		pybind11::class_<LReshape, std::shared_ptr<LReshape>, PyCallBack_LReshape, LinLayer> cl(M(""), "LReshape", "Reshape Layer");
		cl.def( pybind11::init( [](PyCallBack_LReshape const &o){ return new PyCallBack_LReshape(o); } ) );
		cl.def( pybind11::init( [](LReshape const &o){ return new LReshape(o); } ) );
		cl.def_readwrite("ls", &LReshape::ls);
		cl.def("forward", (void (LReshape::*)()) &LReshape::forward, "C++: LReshape::forward() --> void");
		cl.def("backward", (void (LReshape::*)()) &LReshape::backward, "C++: LReshape::backward() --> void");
		cl.def("resize", (void (LReshape::*)(int)) &LReshape::resize, "C++: LReshape::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LReshape & (LReshape::*)(const class LReshape &)) &LReshape::operator=, "C++: LReshape::operator=(const class LReshape &) --> class LReshape &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lreshape_addons(cl);
	}
	{ // LTranspose file:eddl/layers/core/layer_core.h line:184
		pybind11::class_<LTranspose, std::shared_ptr<LTranspose>, PyCallBack_LTranspose, LinLayer> cl(M(""), "LTranspose", "Transpose Layer");
		cl.def( pybind11::init( [](PyCallBack_LTranspose const &o){ return new PyCallBack_LTranspose(o); } ) );
		cl.def( pybind11::init( [](LTranspose const &o){ return new LTranspose(o); } ) );
		cl.def_readwrite("dims", &LTranspose::dims);
		cl.def_readwrite("rdims", &LTranspose::rdims);
		cl.def("forward", (void (LTranspose::*)()) &LTranspose::forward, "C++: LTranspose::forward() --> void");
		cl.def("backward", (void (LTranspose::*)()) &LTranspose::backward, "C++: LTranspose::backward() --> void");
		cl.def("resize", (void (LTranspose::*)(int)) &LTranspose::resize, "C++: LTranspose::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LTranspose & (LTranspose::*)(const class LTranspose &)) &LTranspose::operator=, "C++: LTranspose::operator=(const class LTranspose &) --> class LTranspose &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LDropout file:eddl/layers/core/layer_core.h line:209
		pybind11::class_<LDropout, std::shared_ptr<LDropout>, PyCallBack_LDropout, LinLayer> cl(M(""), "LDropout", "Drop-out Layer");
		cl.def_readwrite("df", &LDropout::df);
		cl.def("forward", (void (LDropout::*)()) &LDropout::forward, "C++: LDropout::forward() --> void");
		cl.def("backward", (void (LDropout::*)()) &LDropout::backward, "C++: LDropout::backward() --> void");
		cl.def("resize", (void (LDropout::*)(int)) &LDropout::resize, "C++: LDropout::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LDropout & (LDropout::*)(const class LDropout &)) &LDropout::operator=, "C++: LDropout::operator=(const class LDropout &) --> class LDropout &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		ldropout_addons(cl);
	}
	{ // LBatchNorm file:eddl/layers/core/layer_core.h line:235
		pybind11::class_<LBatchNorm, std::shared_ptr<LBatchNorm>, PyCallBack_LBatchNorm, LinLayer> cl(M(""), "LBatchNorm", "BatchNormalization Layer");
		cl.def( pybind11::init( [](PyCallBack_LBatchNorm const &o){ return new PyCallBack_LBatchNorm(o); } ) );
		cl.def( pybind11::init( [](LBatchNorm const &o){ return new LBatchNorm(o); } ) );
		cl.def_readwrite("momentum", &LBatchNorm::momentum);
		cl.def_readwrite("epsilon", &LBatchNorm::epsilon);
		cl.def_readwrite("affine", &LBatchNorm::affine);
		cl.def_readwrite("layers", &LBatchNorm::layers);
		cl.def("forward", (void (LBatchNorm::*)()) &LBatchNorm::forward, "C++: LBatchNorm::forward() --> void");
		cl.def("backward", (void (LBatchNorm::*)()) &LBatchNorm::backward, "C++: LBatchNorm::backward() --> void");
		cl.def("resize", (void (LBatchNorm::*)(int)) &LBatchNorm::resize, "C++: LBatchNorm::resize(int) --> void", pybind11::arg("batch"));
		cl.def("reset", (void (LBatchNorm::*)()) &LBatchNorm::reset, "C++: LBatchNorm::reset() --> void");
		cl.def("assign", (class LBatchNorm & (LBatchNorm::*)(const class LBatchNorm &)) &LBatchNorm::operator=, "C++: LBatchNorm::operator=(const class LBatchNorm &) --> class LBatchNorm &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lbatchnorm_addons(cl);
	}
	{ // LConv file:eddl/layers/conv/layer_conv.h line:27
		pybind11::class_<LConv, std::shared_ptr<LConv>, PyCallBack_LConv, LinLayer> cl(M(""), "LConv", "Conv2D Layer");
		cl.def("forward", (void (LConv::*)()) &LConv::forward, "C++: LConv::forward() --> void");
		cl.def("backward", (void (LConv::*)()) &LConv::backward, "C++: LConv::backward() --> void");
		cl.def("resize", (void (LConv::*)(int)) &LConv::resize, "C++: LConv::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LConv & (LConv::*)(const class LConv &)) &LConv::operator=, "C++: LConv::operator=(const class LConv &) --> class LConv &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lconv_addons(cl);
	}
	{ // LConvT file:eddl/layers/conv/layer_conv.h line:60
		pybind11::class_<LConvT, std::shared_ptr<LConvT>, PyCallBack_LConvT, LinLayer> cl(M(""), "LConvT", "ConvT2D Layer");
		cl.def("assign", (class LConvT & (LConvT::*)(const class LConvT &)) &LConvT::operator=, "C++: LConvT::operator=(const class LConvT &) --> class LConvT &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LUpSampling file:eddl/layers/conv/layer_conv.h line:87
		pybind11::class_<LUpSampling, std::shared_ptr<LUpSampling>, PyCallBack_LUpSampling, LinLayer> cl(M(""), "LUpSampling", "UpSampling2D Layer");
		cl.def( pybind11::init( [](PyCallBack_LUpSampling const &o){ return new PyCallBack_LUpSampling(o); } ) );
		cl.def( pybind11::init( [](LUpSampling const &o){ return new LUpSampling(o); } ) );
		cl.def_readwrite("size", &LUpSampling::size);
		cl.def_readwrite("interpolation", &LUpSampling::interpolation);
		cl.def("forward", (void (LUpSampling::*)()) &LUpSampling::forward, "C++: LUpSampling::forward() --> void");
		cl.def("backward", (void (LUpSampling::*)()) &LUpSampling::backward, "C++: LUpSampling::backward() --> void");
		cl.def("resize", (void (LUpSampling::*)(int)) &LUpSampling::resize, "C++: LUpSampling::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LUpSampling & (LUpSampling::*)(const class LUpSampling &)) &LUpSampling::operator=, "C++: LUpSampling::operator=(const class LUpSampling &) --> class LUpSampling &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lupsampling_addons(cl);
	}
	{ // LPool file:eddl/layers/pool/layer_pool.h line:27
		pybind11::class_<LPool, std::shared_ptr<LPool>, PyCallBack_LPool, LinLayer> cl(M(""), "LPool", "Pool2D Layer");
		cl.def("resize", (void (LPool::*)(int)) &LPool::resize, "C++: LPool::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LPool & (LPool::*)(const class LPool &)) &LPool::operator=, "C++: LPool::operator=(const class LPool &) --> class LPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMaxPool file:eddl/layers/pool/layer_pool.h line:39
		pybind11::class_<LMaxPool, std::shared_ptr<LMaxPool>, PyCallBack_LMaxPool, LPool> cl(M(""), "LMaxPool", "MaxPool2D Layer");
		cl.def("forward", (void (LMaxPool::*)()) &LMaxPool::forward, "C++: LMaxPool::forward() --> void");
		cl.def("backward", (void (LMaxPool::*)()) &LMaxPool::backward, "C++: LMaxPool::backward() --> void");
		cl.def("resize", (void (LMaxPool::*)(int)) &LMaxPool::resize, "C++: LMaxPool::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LMaxPool & (LMaxPool::*)(const class LMaxPool &)) &LMaxPool::operator=, "C++: LMaxPool::operator=(const class LMaxPool &) --> class LMaxPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lmaxpool_addons(cl);
	}
	{ // LAveragePool file:eddl/layers/pool/layer_pool.h line:68
		pybind11::class_<LAveragePool, std::shared_ptr<LAveragePool>, PyCallBack_LAveragePool, LPool> cl(M(""), "LAveragePool", "AveragePool2D Layer");
		cl.def("assign", (class LAveragePool & (LAveragePool::*)(const class LAveragePool &)) &LAveragePool::operator=, "C++: LAveragePool::operator=(const class LAveragePool &) --> class LAveragePool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGlobalMaxPool file:eddl/layers/pool/layer_pool.h line:92
		pybind11::class_<LGlobalMaxPool, std::shared_ptr<LGlobalMaxPool>, PyCallBack_LGlobalMaxPool, LPool> cl(M(""), "LGlobalMaxPool", "GlobalMaxPool2D Layer");
		cl.def("assign", (class LGlobalMaxPool & (LGlobalMaxPool::*)(const class LGlobalMaxPool &)) &LGlobalMaxPool::operator=, "C++: LGlobalMaxPool::operator=(const class LGlobalMaxPool &) --> class LGlobalMaxPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGlobalAveragePool file:eddl/layers/pool/layer_pool.h line:115
		pybind11::class_<LGlobalAveragePool, std::shared_ptr<LGlobalAveragePool>, PyCallBack_LGlobalAveragePool, LPool> cl(M(""), "LGlobalAveragePool", "GlobalAveragePool2D Layer");
		cl.def("assign", (class LGlobalAveragePool & (LGlobalAveragePool::*)(const class LGlobalAveragePool &)) &LGlobalAveragePool::operator=, "C++: LGlobalAveragePool::operator=(const class LGlobalAveragePool &) --> class LGlobalAveragePool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGaussianNoise file:eddl/layers/noise/layer_noise.h line:26
		pybind11::class_<LGaussianNoise, std::shared_ptr<LGaussianNoise>, PyCallBack_LGaussianNoise, LinLayer> cl(M(""), "LGaussianNoise", "GaussianNoise Layer");
		cl.def_readwrite("stdev", &LGaussianNoise::stdev);
		cl.def("resize", (void (LGaussianNoise::*)(int)) &LGaussianNoise::resize, "C++: LGaussianNoise::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (LGaussianNoise::*)()) &LGaussianNoise::forward, "C++: LGaussianNoise::forward() --> void");
		cl.def("backward", (void (LGaussianNoise::*)()) &LGaussianNoise::backward, "C++: LGaussianNoise::backward() --> void");
		cl.def("assign", (class LGaussianNoise & (LGaussianNoise::*)(const class LGaussianNoise &)) &LGaussianNoise::operator=, "C++: LGaussianNoise::operator=(const class LGaussianNoise &) --> class LGaussianNoise &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lgaussiannoise_addons(cl);
	}
	{ // LAdd file:eddl/layers/merge/layer_merge.h line:26
		pybind11::class_<LAdd, std::shared_ptr<LAdd>, PyCallBack_LAdd, MLayer> cl(M(""), "LAdd", "Add Layer");
		cl.def("forward", (void (LAdd::*)()) &LAdd::forward, "C++: LAdd::forward() --> void");
		cl.def("backward", (void (LAdd::*)()) &LAdd::backward, "C++: LAdd::backward() --> void");
		cl.def("resize", (void (LAdd::*)(int)) &LAdd::resize, "C++: LAdd::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LAdd & (LAdd::*)(const class LAdd &)) &LAdd::operator=, "C++: LAdd::operator=(const class LAdd &) --> class LAdd &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LSubtract file:eddl/layers/merge/layer_merge.h line:48
		pybind11::class_<LSubtract, std::shared_ptr<LSubtract>, PyCallBack_LSubtract, MLayer> cl(M(""), "LSubtract", "Subtract Layer");
		cl.def("forward", (void (LSubtract::*)()) &LSubtract::forward, "C++: LSubtract::forward() --> void");
		cl.def("backward", (void (LSubtract::*)()) &LSubtract::backward, "C++: LSubtract::backward() --> void");
		cl.def("assign", (class LSubtract & (LSubtract::*)(const class LSubtract &)) &LSubtract::operator=, "C++: LSubtract::operator=(const class LSubtract &) --> class LSubtract &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMatMul file:eddl/layers/merge/layer_merge.h line:68
		pybind11::class_<LMatMul, std::shared_ptr<LMatMul>, PyCallBack_LMatMul, MLayer> cl(M(""), "LMatMul", "MatMul Layer");
		cl.def("forward", (void (LMatMul::*)()) &LMatMul::forward, "C++: LMatMul::forward() --> void");
		cl.def("backward", (void (LMatMul::*)()) &LMatMul::backward, "C++: LMatMul::backward() --> void");
		cl.def("assign", (class LMatMul & (LMatMul::*)(const class LMatMul &)) &LMatMul::operator=, "C++: LMatMul::operator=(const class LMatMul &) --> class LMatMul &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LAverage file:eddl/layers/merge/layer_merge.h line:88
		pybind11::class_<LAverage, std::shared_ptr<LAverage>, PyCallBack_LAverage, MLayer> cl(M(""), "LAverage", "Average Layer");
		cl.def("forward", (void (LAverage::*)()) &LAverage::forward, "C++: LAverage::forward() --> void");
		cl.def("backward", (void (LAverage::*)()) &LAverage::backward, "C++: LAverage::backward() --> void");
		cl.def("resize", (void (LAverage::*)(int)) &LAverage::resize, "C++: LAverage::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LAverage & (LAverage::*)(const class LAverage &)) &LAverage::operator=, "C++: LAverage::operator=(const class LAverage &) --> class LAverage &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMaximum file:eddl/layers/merge/layer_merge.h line:110
		pybind11::class_<LMaximum, std::shared_ptr<LMaximum>, PyCallBack_LMaximum, MLayer> cl(M(""), "LMaximum", "Maximum Layer");
		cl.def("forward", (void (LMaximum::*)()) &LMaximum::forward, "C++: LMaximum::forward() --> void");
		cl.def("backward", (void (LMaximum::*)()) &LMaximum::backward, "C++: LMaximum::backward() --> void");
		cl.def("assign", (class LMaximum & (LMaximum::*)(const class LMaximum &)) &LMaximum::operator=, "C++: LMaximum::operator=(const class LMaximum &) --> class LMaximum &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMinimum file:eddl/layers/merge/layer_merge.h line:130
		pybind11::class_<LMinimum, std::shared_ptr<LMinimum>, PyCallBack_LMinimum, MLayer> cl(M(""), "LMinimum", "Maximum Layer");
		cl.def("forward", (void (LMinimum::*)()) &LMinimum::forward, "C++: LMinimum::forward() --> void");
		cl.def("backward", (void (LMinimum::*)()) &LMinimum::backward, "C++: LMinimum::backward() --> void");
		cl.def("assign", (class LMinimum & (LMinimum::*)(const class LMinimum &)) &LMinimum::operator=, "C++: LMinimum::operator=(const class LMinimum &) --> class LMinimum &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LConcat file:eddl/layers/merge/layer_merge.h line:150
		pybind11::class_<LConcat, std::shared_ptr<LConcat>, PyCallBack_LConcat, MLayer> cl(M(""), "LConcat", "Concat Layer");
		cl.def( pybind11::init( [](PyCallBack_LConcat const &o){ return new PyCallBack_LConcat(o); } ) );
		cl.def( pybind11::init( [](LConcat const &o){ return new LConcat(o); } ) );
		cl.def_readwrite("ndim", &LConcat::ndim);
		cl.def_readwrite("index", &LConcat::index);
		cl.def("forward", (void (LConcat::*)()) &LConcat::forward, "C++: LConcat::forward() --> void");
		cl.def("backward", (void (LConcat::*)()) &LConcat::backward, "C++: LConcat::backward() --> void");
		cl.def("resize", (void (LConcat::*)(int)) &LConcat::resize, "C++: LConcat::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LConcat & (LConcat::*)(const class LConcat &)) &LConcat::operator=, "C++: LConcat::operator=(const class LConcat &) --> class LConcat &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lconcat_addons(cl);
	}
	{ // ReductionLayer file:eddl/layers/reductions/layer_reductions.h line:30
		pybind11::class_<ReductionLayer, std::shared_ptr<ReductionLayer>, PyCallBack_ReductionLayer, Layer> cl(M(""), "ReductionLayer", "//////////////////////////////////////\n//////////////////////////////////////");
		cl.def( pybind11::init( [](PyCallBack_ReductionLayer const &o){ return new PyCallBack_ReductionLayer(o); } ) );
		cl.def( pybind11::init( [](ReductionLayer const &o){ return new ReductionLayer(o); } ) );
		cl.def_readwrite("binary", &ReductionLayer::binary);
		cl.def_readwrite("val", &ReductionLayer::val);
		cl.def("addchild", (void (ReductionLayer::*)(class Layer *)) &ReductionLayer::addchild, "C++: ReductionLayer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (ReductionLayer::*)(class Layer *)) &ReductionLayer::addparent, "C++: ReductionLayer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("assign", (class ReductionLayer & (ReductionLayer::*)(const class ReductionLayer &)) &ReductionLayer::operator=, "C++: ReductionLayer::operator=(const class ReductionLayer &) --> class ReductionLayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/layers/reductions/layer_reductions.cpp
#include <eddl/initializers/initializer.h>
#include <eddl/layers/layer.h>
#include <eddl/layers/reductions/layer_reductions.h>
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <lrmean_addons.hpp>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// LRMean file:eddl/layers/reductions/layer_reductions.h line:50
struct PyCallBack_LRMean : public LRMean {
	using LRMean::LRMean;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMean::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMean::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMean::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMean *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LRVar file:eddl/layers/reductions/layer_reductions.h line:69
struct PyCallBack_LRVar : public LRVar {
	using LRVar::LRVar;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRVar::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRVar::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRVar::resize(a0);
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRVar::reset();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRVar *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
};

// LRSum file:eddl/layers/reductions/layer_reductions.h line:94
struct PyCallBack_LRSum : public LRSum {
	using LRSum::LRSum;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRSum::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRSum::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRSum::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRSum *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LRMax file:eddl/layers/reductions/layer_reductions.h line:112
struct PyCallBack_LRMax : public LRMax {
	using LRMax::LRMax;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMax::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMax::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMax::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMax *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// LRMin file:eddl/layers/reductions/layer_reductions.h line:131
struct PyCallBack_LRMin : public LRMin {
	using LRMin::LRMin;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMin::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMin::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LRMin::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReductionLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LRMin *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
};

// Optimizer file:eddl/optimizers/optim.h line:27
struct PyCallBack_Optimizer : public Optimizer {
	using Optimizer::Optimizer;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Optimizer *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Optimizer *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// SGD file:eddl/optimizers/optim.h line:44
struct PyCallBack_SGD : public SGD {
	using SGD::SGD;

	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SGD *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return SGD::clone();
	}
	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SGD *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SGD::applygrads(a0);
	}
};

// Adam file:eddl/optimizers/optim.h line:65
struct PyCallBack_Adam : public Adam {
	using Adam::Adam;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adam *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adam *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// AdaDelta file:eddl/optimizers/optim.h line:89
struct PyCallBack_AdaDelta : public AdaDelta {
	using AdaDelta::AdaDelta;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const AdaDelta *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const AdaDelta *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// Adagrad file:eddl/optimizers/optim.h line:110
struct PyCallBack_Adagrad : public Adagrad {
	using Adagrad::Adagrad;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adagrad *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adagrad *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// Adamax file:eddl/optimizers/optim.h line:130
struct PyCallBack_Adamax : public Adamax {
	using Adamax::Adamax;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adamax *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adamax *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// Nadam file:eddl/optimizers/optim.h line:152
struct PyCallBack_Nadam : public Nadam {
	using Nadam::Nadam;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Nadam *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Nadam *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

// RMSProp file:eddl/optimizers/optim.h line:174
struct PyCallBack_RMSProp : public RMSProp {
	using RMSProp::RMSProp;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RMSProp *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RMSProp *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

void bind_eddl_layers_reductions_layer_reductions(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // LRMean file:eddl/layers/reductions/layer_reductions.h line:50
		pybind11::class_<LRMean, std::shared_ptr<LRMean>, PyCallBack_LRMean, ReductionLayer> cl(M(""), "LRMean", "Mean Layer");
		cl.def("forward", (void (LRMean::*)()) &LRMean::forward, "C++: LRMean::forward() --> void");
		cl.def("backward", (void (LRMean::*)()) &LRMean::backward, "C++: LRMean::backward() --> void");
		cl.def("resize", (void (LRMean::*)(int)) &LRMean::resize, "C++: LRMean::resize(int) --> void", pybind11::arg("b"));
		cl.def("assign", (class LRMean & (LRMean::*)(const class LRMean &)) &LRMean::operator=, "C++: LRMean::operator=(const class LRMean &) --> class LRMean &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lrmean_addons(cl);
	}
	{ // LRVar file:eddl/layers/reductions/layer_reductions.h line:69
		pybind11::class_<LRVar, std::shared_ptr<LRVar>, PyCallBack_LRVar, ReductionLayer> cl(M(""), "LRVar", "Var Layer");
		cl.def( pybind11::init( [](PyCallBack_LRVar const &o){ return new PyCallBack_LRVar(o); } ) );
		cl.def( pybind11::init( [](LRVar const &o){ return new LRVar(o); } ) );
		cl.def_readwrite("axis", &LRVar::axis);
		cl.def_readwrite("keepdims", &LRVar::keepdims);
		cl.def_readwrite("layers", &LRVar::layers);
		cl.def("forward", (void (LRVar::*)()) &LRVar::forward, "C++: LRVar::forward() --> void");
		cl.def("backward", (void (LRVar::*)()) &LRVar::backward, "C++: LRVar::backward() --> void");
		cl.def("resize", (void (LRVar::*)(int)) &LRVar::resize, "C++: LRVar::resize(int) --> void", pybind11::arg("b"));
		cl.def("reset", (void (LRVar::*)()) &LRVar::reset, "C++: LRVar::reset() --> void");
		cl.def("assign", (class LRVar & (LRVar::*)(const class LRVar &)) &LRVar::operator=, "C++: LRVar::operator=(const class LRVar &) --> class LRVar &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LRSum file:eddl/layers/reductions/layer_reductions.h line:94
		pybind11::class_<LRSum, std::shared_ptr<LRSum>, PyCallBack_LRSum, ReductionLayer> cl(M(""), "LRSum", "Sum Layer");
		cl.def("forward", (void (LRSum::*)()) &LRSum::forward, "C++: LRSum::forward() --> void");
		cl.def("backward", (void (LRSum::*)()) &LRSum::backward, "C++: LRSum::backward() --> void");
		cl.def("resize", (void (LRSum::*)(int)) &LRSum::resize, "C++: LRSum::resize(int) --> void", pybind11::arg("b"));
		cl.def("assign", (class LRSum & (LRSum::*)(const class LRSum &)) &LRSum::operator=, "C++: LRSum::operator=(const class LRSum &) --> class LRSum &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LRMax file:eddl/layers/reductions/layer_reductions.h line:112
		pybind11::class_<LRMax, std::shared_ptr<LRMax>, PyCallBack_LRMax, ReductionLayer> cl(M(""), "LRMax", "Max Layer");
		cl.def("forward", (void (LRMax::*)()) &LRMax::forward, "C++: LRMax::forward() --> void");
		cl.def("backward", (void (LRMax::*)()) &LRMax::backward, "C++: LRMax::backward() --> void");
		cl.def("resize", (void (LRMax::*)(int)) &LRMax::resize, "C++: LRMax::resize(int) --> void", pybind11::arg("b"));
		cl.def("assign", (class LRMax & (LRMax::*)(const class LRMax &)) &LRMax::operator=, "C++: LRMax::operator=(const class LRMax &) --> class LRMax &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LRMin file:eddl/layers/reductions/layer_reductions.h line:131
		pybind11::class_<LRMin, std::shared_ptr<LRMin>, PyCallBack_LRMin, ReductionLayer> cl(M(""), "LRMin", "Min Layer");
		cl.def("forward", (void (LRMin::*)()) &LRMin::forward, "C++: LRMin::forward() --> void");
		cl.def("backward", (void (LRMin::*)()) &LRMin::backward, "C++: LRMin::backward() --> void");
		cl.def("resize", (void (LRMin::*)(int)) &LRMin::resize, "C++: LRMin::resize(int) --> void", pybind11::arg("b"));
		cl.def("assign", (class LRMin & (LRMin::*)(const class LRMin &)) &LRMin::operator=, "C++: LRMin::operator=(const class LRMin &) --> class LRMin &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Optimizer file:eddl/optimizers/optim.h line:27
		pybind11::class_<Optimizer, std::shared_ptr<Optimizer>, PyCallBack_Optimizer> cl(M(""), "Optimizer", "");
		cl.def( pybind11::init( [](){ return new Optimizer(); }, [](){ return new PyCallBack_Optimizer(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Optimizer const &o){ return new PyCallBack_Optimizer(o); } ) );
		cl.def( pybind11::init( [](Optimizer const &o){ return new Optimizer(o); } ) );
		cl.def_readwrite("name", &Optimizer::name);
		cl.def_readwrite("layers", &Optimizer::layers);
		cl.def("applygrads", (void (Optimizer::*)(int)) &Optimizer::applygrads, "C++: Optimizer::applygrads(int) --> void", pybind11::arg("batch"));
		cl.def("clone", (class Optimizer * (Optimizer::*)()) &Optimizer::clone, "C++: Optimizer::clone() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("assign", (class Optimizer & (Optimizer::*)(const class Optimizer &)) &Optimizer::operator=, "C++: Optimizer::operator=(const class Optimizer &) --> class Optimizer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // SGD file:eddl/optimizers/optim.h line:44
		pybind11::class_<SGD, std::shared_ptr<SGD>, PyCallBack_SGD, Optimizer> cl(M(""), "SGD", "");
		cl.def( pybind11::init( [](){ return new SGD(); }, [](){ return new PyCallBack_SGD(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new SGD(a0); }, [](float const & a0){ return new PyCallBack_SGD(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new SGD(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_SGD(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new SGD(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_SGD(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, bool>(), pybind11::arg("lr"), pybind11::arg("momentum"), pybind11::arg("weight_decay"), pybind11::arg("nesterov") );

		cl.def( pybind11::init( [](PyCallBack_SGD const &o){ return new PyCallBack_SGD(o); } ) );
		cl.def( pybind11::init( [](SGD const &o){ return new SGD(o); } ) );
		cl.def_readwrite("lr", &SGD::lr);
		cl.def_readwrite("mu", &SGD::mu);
		cl.def_readwrite("weight_decay", &SGD::weight_decay);
		cl.def_readwrite("nesterov", &SGD::nesterov);
		cl.def_readwrite("mT", &SGD::mT);
		cl.def("clone", (class Optimizer * (SGD::*)()) &SGD::clone, "C++: SGD::clone() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("applygrads", (void (SGD::*)(int)) &SGD::applygrads, "C++: SGD::applygrads(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class SGD & (SGD::*)(const class SGD &)) &SGD::operator=, "C++: SGD::operator=(const class SGD &) --> class SGD &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Adam file:eddl/optimizers/optim.h line:65
		pybind11::class_<Adam, std::shared_ptr<Adam>, PyCallBack_Adam, Optimizer> cl(M(""), "Adam", "");
		cl.def( pybind11::init( [](){ return new Adam(); }, [](){ return new PyCallBack_Adam(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adam(a0); }, [](float const & a0){ return new PyCallBack_Adam(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adam(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adam(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Adam(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Adam(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Adam(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Adam(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3, float const & a4){ return new Adam(a0, a1, a2, a3, a4); }, [](float const & a0, float const & a1, float const & a2, float const & a3, float const & a4){ return new PyCallBack_Adam(a0, a1, a2, a3, a4); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float, bool>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("weight_decay"), pybind11::arg("amsgrad") );

		cl.def( pybind11::init( [](PyCallBack_Adam const &o){ return new PyCallBack_Adam(o); } ) );
		cl.def( pybind11::init( [](Adam const &o){ return new Adam(o); } ) );
		cl.def_readwrite("lr", &Adam::lr);
		cl.def_readwrite("beta_1", &Adam::beta_1);
		cl.def_readwrite("beta_2", &Adam::beta_2);
		cl.def_readwrite("epsilon", &Adam::epsilon);
		cl.def_readwrite("weight_decay", &Adam::weight_decay);
		cl.def_readwrite("amsgrad", &Adam::amsgrad);
		cl.def_readwrite("mT", &Adam::mT);
		cl.def("assign", (class Adam & (Adam::*)(const class Adam &)) &Adam::operator=, "C++: Adam::operator=(const class Adam &) --> class Adam &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // AdaDelta file:eddl/optimizers/optim.h line:89
		pybind11::class_<AdaDelta, std::shared_ptr<AdaDelta>, PyCallBack_AdaDelta, Optimizer> cl(M(""), "AdaDelta", "");
		cl.def( pybind11::init( [](){ return new AdaDelta(); }, [](){ return new PyCallBack_AdaDelta(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new AdaDelta(a0); }, [](float const & a0){ return new PyCallBack_AdaDelta(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new AdaDelta(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_AdaDelta(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new AdaDelta(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_AdaDelta(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("rho"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def( pybind11::init( [](PyCallBack_AdaDelta const &o){ return new PyCallBack_AdaDelta(o); } ) );
		cl.def( pybind11::init( [](AdaDelta const &o){ return new AdaDelta(o); } ) );
		cl.def_readwrite("lr", &AdaDelta::lr);
		cl.def_readwrite("rho", &AdaDelta::rho);
		cl.def_readwrite("epsilon", &AdaDelta::epsilon);
		cl.def_readwrite("weight_decay", &AdaDelta::weight_decay);
		cl.def_readwrite("mT", &AdaDelta::mT);
		cl.def("assign", (class AdaDelta & (AdaDelta::*)(const class AdaDelta &)) &AdaDelta::operator=, "C++: AdaDelta::operator=(const class AdaDelta &) --> class AdaDelta &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Adagrad file:eddl/optimizers/optim.h line:110
		pybind11::class_<Adagrad, std::shared_ptr<Adagrad>, PyCallBack_Adagrad, Optimizer> cl(M(""), "Adagrad", "");
		cl.def( pybind11::init( [](){ return new Adagrad(); }, [](){ return new PyCallBack_Adagrad(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adagrad(a0); }, [](float const & a0){ return new PyCallBack_Adagrad(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adagrad(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adagrad(a0, a1); } ), "doc");
		cl.def( pybind11::init<float, float, float>(), pybind11::arg("lr"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def( pybind11::init( [](PyCallBack_Adagrad const &o){ return new PyCallBack_Adagrad(o); } ) );
		cl.def( pybind11::init( [](Adagrad const &o){ return new Adagrad(o); } ) );
		cl.def_readwrite("lr", &Adagrad::lr);
		cl.def_readwrite("epsilon", &Adagrad::epsilon);
		cl.def_readwrite("weight_decay", &Adagrad::weight_decay);
		cl.def_readwrite("mT", &Adagrad::mT);
		cl.def("assign", (class Adagrad & (Adagrad::*)(const class Adagrad &)) &Adagrad::operator=, "C++: Adagrad::operator=(const class Adagrad &) --> class Adagrad &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Adamax file:eddl/optimizers/optim.h line:130
		pybind11::class_<Adamax, std::shared_ptr<Adamax>, PyCallBack_Adamax, Optimizer> cl(M(""), "Adamax", "");
		cl.def( pybind11::init( [](){ return new Adamax(); }, [](){ return new PyCallBack_Adamax(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adamax(a0); }, [](float const & a0){ return new PyCallBack_Adamax(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adamax(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adamax(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Adamax(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Adamax(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Adamax(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Adamax(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def( pybind11::init( [](PyCallBack_Adamax const &o){ return new PyCallBack_Adamax(o); } ) );
		cl.def( pybind11::init( [](Adamax const &o){ return new Adamax(o); } ) );
		cl.def_readwrite("lr", &Adamax::lr);
		cl.def_readwrite("beta_1", &Adamax::beta_1);
		cl.def_readwrite("beta_2", &Adamax::beta_2);
		cl.def_readwrite("epsilon", &Adamax::epsilon);
		cl.def_readwrite("weight_decay", &Adamax::weight_decay);
		cl.def_readwrite("mT", &Adamax::mT);
		cl.def("assign", (class Adamax & (Adamax::*)(const class Adamax &)) &Adamax::operator=, "C++: Adamax::operator=(const class Adamax &) --> class Adamax &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Nadam file:eddl/optimizers/optim.h line:152
		pybind11::class_<Nadam, std::shared_ptr<Nadam>, PyCallBack_Nadam, Optimizer> cl(M(""), "Nadam", "");
		cl.def( pybind11::init( [](){ return new Nadam(); }, [](){ return new PyCallBack_Nadam(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Nadam(a0); }, [](float const & a0){ return new PyCallBack_Nadam(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Nadam(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Nadam(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Nadam(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Nadam(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Nadam(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Nadam(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("schedule_decay") );

		cl.def( pybind11::init( [](PyCallBack_Nadam const &o){ return new PyCallBack_Nadam(o); } ) );
		cl.def( pybind11::init( [](Nadam const &o){ return new Nadam(o); } ) );
		cl.def_readwrite("lr", &Nadam::lr);
		cl.def_readwrite("beta_1", &Nadam::beta_1);
		cl.def_readwrite("beta_2", &Nadam::beta_2);
		cl.def_readwrite("epsilon", &Nadam::epsilon);
		cl.def_readwrite("schedule_decay", &Nadam::schedule_decay);
		cl.def_readwrite("mT", &Nadam::mT);
		cl.def("assign", (class Nadam & (Nadam::*)(const class Nadam &)) &Nadam::operator=, "C++: Nadam::operator=(const class Nadam &) --> class Nadam &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // RMSProp file:eddl/optimizers/optim.h line:174
		pybind11::class_<RMSProp, std::shared_ptr<RMSProp>, PyCallBack_RMSProp, Optimizer> cl(M(""), "RMSProp", "");
		cl.def( pybind11::init( [](){ return new RMSProp(); }, [](){ return new PyCallBack_RMSProp(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new RMSProp(a0); }, [](float const & a0){ return new PyCallBack_RMSProp(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new RMSProp(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_RMSProp(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new RMSProp(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_RMSProp(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("rho"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def( pybind11::init( [](PyCallBack_RMSProp const &o){ return new PyCallBack_RMSProp(o); } ) );
		cl.def( pybind11::init( [](RMSProp const &o){ return new RMSProp(o); } ) );
		cl.def_readwrite("lr", &RMSProp::lr);
		cl.def_readwrite("rho", &RMSProp::rho);
		cl.def_readwrite("epsilon", &RMSProp::epsilon);
		cl.def_readwrite("weight_decay", &RMSProp::weight_decay);
		cl.def_readwrite("mT", &RMSProp::mT);
		cl.def("assign", (class RMSProp & (RMSProp::*)(const class RMSProp &)) &RMSProp::operator=, "C++: RMSProp::operator=(const class RMSProp &) --> class RMSProp &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/net.cpp
#include <custom_binder.hpp>
#include <dummy.hpp>
#include <eddl/compserv.h>
#include <eddl/initializers/initializer.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/net.h>
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <memory>
#include <net_addons.hpp>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_net(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// train_batch_t(void *) file:eddl/net.h line:34
	M("").def("train_batch_t", (void * (*)(void *)) &train_batch_t, "C++: train_batch_t(void *) --> void *", pybind11::return_value_policy::automatic, pybind11::arg("targs"));

	{ // Net file:eddl/net.h line:39
		pybind11::class_<Net, std::shared_ptr<Net>> cl(M(""), "Net", "");
		cl.def( pybind11::init( [](Net const &o){ return new Net(o); } ) );
		cl.def_readwrite("name", &Net::name);
		cl.def_readwrite("dev", &Net::dev);
		cl.def_readwrite("batch_size", &Net::batch_size);
		cl.def_readwrite("tr_batches", &Net::tr_batches);
		cl.def_readwrite("devsel", &Net::devsel);
		cl.def_readwrite("layers", &Net::layers);
		cl.def_readwrite("lin", &Net::lin);
		cl.def_readwrite("lout", &Net::lout);
		cl.def_readwrite("vfts", &Net::vfts);
		cl.def_readwrite("vbts", &Net::vbts);
		cl.def_readwrite("losses", &Net::losses);
		cl.def_readwrite("metrics", &Net::metrics);
		cl.def_readwrite("fiterr", &Net::fiterr);
		cl.def_readwrite("snets", &Net::snets);
		cl.def("initialize", (void (Net::*)()) &Net::initialize, "C++: Net::initialize() --> void");
		cl.def("reset", (void (Net::*)()) &Net::reset, "C++: Net::reset() --> void");
		cl.def("forward", (void (Net::*)()) &Net::forward, "C++: Net::forward() --> void");
		cl.def("delta", (void (Net::*)()) &Net::delta, "C++: Net::delta() --> void");
		cl.def("loss", (void (Net::*)()) &Net::loss, "C++: Net::loss() --> void");
		cl.def("backward", (void (Net::*)()) &Net::backward, "C++: Net::backward() --> void");
		cl.def("applygrads", (void (Net::*)()) &Net::applygrads, "C++: Net::applygrads() --> void");
		cl.def("split", (void (Net::*)(int, int)) &Net::split, "C++: Net::split(int, int) --> void", pybind11::arg("c"), pybind11::arg("todev"));
		cl.def("inNet", (int (Net::*)(class Layer *)) &Net::inNet, "C++: Net::inNet(class Layer *) --> int", pybind11::arg("l"));
		cl.def("walk", (void (Net::*)(class Layer *)) &Net::walk, "C++: Net::walk(class Layer *) --> void", pybind11::arg("l"));
		cl.def("walk_back", (void (Net::*)(class Layer *)) &Net::walk_back, "C++: Net::walk_back(class Layer *) --> void", pybind11::arg("l"));
		cl.def("fts", (void (Net::*)()) &Net::fts, "C++: Net::fts() --> void");
		cl.def("bts", (void (Net::*)()) &Net::bts, "C++: Net::bts() --> void");
		cl.def("resize", (void (Net::*)(int)) &Net::resize, "C++: Net::resize(int) --> void", pybind11::arg("batch"));
		cl.def("setmode", (void (Net::*)(int)) &Net::setmode, "C++: Net::setmode(int) --> void", pybind11::arg("m"));
		cl.def("sync_weights", (void (Net::*)()) &Net::sync_weights, "C++: Net::sync_weights() --> void");
		cl.def("clean_fiterr", (void (Net::*)()) &Net::clean_fiterr, "C++: Net::clean_fiterr() --> void");

		net_addons(cl);
	}
	// Dummy file:dummy.hpp line:3
	bind_custom_metric<pybind11::module>(M(""));

}


#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_bits_libio(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_descriptors_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_initializers_initializer(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_layers_core_layer_core(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_layers_reductions_layer_reductions(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_net(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(_core, root_module) {
	root_module.doc() = "_core module";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_);
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	std::vector< std::pair<std::string, std::string> > sub_modules {
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule(p.second.c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_bits_libio(M);
	bind_eddl_tensor_tensor(M);
	bind_eddl_descriptors_descriptors(M);
	bind_eddl_initializers_initializer(M);
	bind_eddl_layers_core_layer_core(M);
	bind_eddl_layers_reductions_layer_reductions(M);
	bind_eddl_net(M);

}

// Source list file: /pyeddl/codegen/bindings/_core.sources
// _core.cpp
// bits/libio.cpp
// eddl/tensor/tensor.cpp
// eddl/descriptors/descriptors.cpp
// eddl/initializers/initializer.cpp
// eddl/layers/core/layer_core.cpp
// eddl/layers/reductions/layer_reductions.cpp
// eddl/net.cpp

// Modules list file: /pyeddl/codegen/bindings/_core.modules
// 
