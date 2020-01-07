// Copyright (c) 2019 CRS4
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <pybind11/pybind11.h>
#include <eddl/metrics/metric.h>


class CustomMetric : public Metric {
public:
    pybind11::object pymetric;
    CustomMetric(pybind11::object pymetric, std::string name);
    float value(Tensor *T, Tensor *Y) override;
};

CustomMetric::CustomMetric(pybind11::object pymetric, std::string name) :
  Metric(name) {
    this->pymetric = pymetric;
}

float CustomMetric::value(Tensor *T, Tensor *Y) {
    pybind11::object pyvalue = pymetric(pybind11::cast(T), pybind11::cast(Y));
    return pyvalue.cast<float>();
}


CustomMetric* getCustomMetric(pybind11::object pymetric, std::string name) {
    return new CustomMetric(pymetric, name);
}


template<typename Module>
void bind_custom(Module &m) {
    pybind11::class_<CustomMetric, std::shared_ptr<CustomMetric>, Metric>
      cl(m, "CustomMetric");
    cl.def(pybind11::init<pybind11::object, std::string>());
    cl.def("value", &CustomMetric::value);
    m.def("getCustomMetric", &getCustomMetric, "getCustomMetric(pymetric, name) --> CustomMetric", pybind11::return_value_policy::reference, pybind11::keep_alive<0, 1>());
}
