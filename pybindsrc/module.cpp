/**
 * @file module.cpp
 *
 * This is part of the DUNE DAQ Software Suite, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dtpfeedbacktools/RawFileReader.hpp"


namespace py = pybind11;

namespace dunedaq
{
  namespace dtpfeedbacktools
  {
    namespace python
    {

      PYBIND11_MODULE(_daq_dtpfeedbacktools_py, m)
      {
        m.doc() = "c++ implementation of the dunedaq dtp feedbac tools modules"; // optional module docstring

        py::class_<RawDataBlock>(m, "RawDataBlock")
            .def("data", &RawDataBlock::data)
            .def("size", &RawDataBlock::size)
            .def(
                "get_capsule",
                [](RawDataBlock &self) -> void*
                { return static_cast<void *>(self.data()); },
                py::return_value_policy::reference_internal);
        ;

        py::class_<RawFileReader>(m, "RawFileReader")
            .def(py::init<const std::string &>())
            .def("get_size", &RawFileReader::get_size)
            // .def("read_block", &RawFileReader::read_block);
            .def("read_block", &RawFileReader::read_block, py::arg("size"), py::arg("offset") = 0)
            ;
      }

    } // namespace python
  }   // namespace dtpfeedbacktools
} // namespace dunedaq
