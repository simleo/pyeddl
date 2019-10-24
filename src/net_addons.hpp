#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void net_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<std::vector<Layer*>, std::vector<Layer*>>(),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def("summary", (std::string (Net::*)()) &Net::summary,
           "C++: Net::summary() --> string");
    cl.def("build", (void (Net::*)(Optimizer*, vloss, vmetrics, CompServ*,
                                   Initializer*)) &Net::build,
           "C++: Net::build(Optimizer*, vloss, vmetrics, CompServ*, Initializer*) --> void",
           pybind11::arg("opt"), pybind11::arg("lo"), pybind11::arg("me"),
           pybind11::arg("cs"), pybind11::arg("init"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>(),
           pybind11::keep_alive<1, 4>(), pybind11::keep_alive<1, 5>(),
           pybind11::keep_alive<1, 6>());
    cl.def("fit", (void (Net::*)(vtensor, vtensor, int, int)) &Net::fit,
           "C++: Net::fit(vtensor, vtensor, int, int) --> void",
           pybind11::arg("tin"), pybind11::arg("tout"),
           pybind11::arg("batch_size"), pybind11::arg("epochs"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def("train_batch",
           (void (Net::*)(vtensor, vtensor, vind, int)) &Net::train_batch,
           "C++: Net::train_batch(vtensor, vtensor, vind, int) --> void",
           pybind11::arg("X"), pybind11::arg("Y"),
           pybind11::arg("sind"), pybind11::arg("eval") = 0,
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def("evaluate", (void (Net::*)(vtensor, vtensor)) &Net::evaluate,
           "C++: Net::evaluate(vtensor, vtensor) --> void",
           pybind11::arg("tin"), pybind11::arg("tout"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def("predict", (void (Net::*)(vtensor, vtensor)) &Net::predict,
           "C++: Net::predict(vtensor, vtensor) --> void",
           pybind11::arg("tin"), pybind11::arg("tout"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def("save", [](Net &net, pybind11::object file_obj) {
      int fd;
      FILE *fp;
      if ((fd = PyObject_AsFileDescriptor(file_obj.ptr())) == -1) {
        throw std::runtime_error("can't convert object to file descriptor");
      }
      if (!(fp = fdopen(fd, "w"))) {
        throw std::runtime_error("failed to open file descriptor");
      }
      net.save(fp);
    });
    cl.def("load", [](Net &net, pybind11::object file_obj) {
      int fd;
      FILE *fp;
      if ((fd = PyObject_AsFileDescriptor(file_obj.ptr())) == -1) {
        throw std::runtime_error("can't convert object to file descriptor");
      }
      if (!(fp = fdopen(fd, "r"))) {
        throw std::runtime_error("failed to open file descriptor");
      }
      net.load(fp);
    });
}
