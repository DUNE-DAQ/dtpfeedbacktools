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
#include "dtpfeedbacktools/FWTP.hpp"

namespace py = pybind11;

namespace dunedaq
{
  namespace dtpfeedbacktools
  {
    namespace python
    {

      std::vector<FWTP> unpack_fwtps(void *buf, size_t n_blocks){
          std::vector<FWTP> fwtps;

          FWTPHeader *hdr = static_cast<FWTPHeader *>(buf);
          FWTPTrailer *trl = static_cast<FWTPTrailer *>(buf);
          // FWTPData *hit = static_cast<FWTPData *>(buf);

          size_t i_hdr = 0;
          size_t i_trl = 0;
      
          for (size_t i(0); i<n_blocks; ++i) {

            if (trl[i].padding_1 != 0xf00d)
            {
              continue;
            }

            i_trl = i;
            fwtps.push_back(FWTP((void*)(hdr+i_hdr), i_trl-(i_hdr+1)));

            i_hdr = i+1;
          }

          return fwtps;
      }

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

        py::class_<FWTPHeader>(m, "FWTPHeader")
          .def_property_readonly("wire_no", [](FWTPHeader &self) -> uint32_t{ return self.wire_no; })
          .def_property_readonly("slot_no", [](FWTPHeader &self) -> uint32_t{ return self.slot_no; })
          .def_property_readonly("flags", [](FWTPHeader &self) -> uint32_t{ return self.flags; })
          .def_property_readonly("crate_no", [](FWTPHeader &self) -> uint32_t{ return self.crate_no; })
          .def_property_readonly("fiber_no", [](FWTPHeader &self) -> uint32_t{ return self.fiber_no; })
          .def_property_readonly("timestamp_1", [](FWTPHeader &self) -> uint32_t{ return self.timestamp_1; })
          .def_property_readonly("timestamp_2", [](FWTPHeader &self) -> uint32_t{ return self.timestamp_2; })
          .def("get_timestamp", &FWTPHeader::get_timestamp)
        ;

        py::class_<FWTPData>(m, "FWTPData")
            .def_property_readonly("end_time", [](FWTPData &self) -> uint32_t { return self.end_time; })
            .def_property_readonly("start_time", [](FWTPData &self) -> uint32_t { return self.start_time; })
            .def_property_readonly("peak_time", [](FWTPData &self) -> uint32_t { return self.peak_time; })
            .def_property_readonly("peak_adc", [](FWTPData &self) -> uint32_t { return self.peak_adc; })
            .def_property_readonly("hit_continue", [](FWTPData &self) -> uint32_t { return self.hit_continue; })
            .def_property_readonly("tp_flags", [](FWTPData &self) -> uint32_t { return self.tp_flags; })
            .def_property_readonly("sum_adc", [](FWTPData &self) -> uint32_t { return self.sum_adc; })
            ;

        py::class_<FWTPTrailer>(m, "FWTPTrailer")
            .def_property_readonly("accumulator", [](FWTPTrailer &self) -> uint32_t { return self.accumulator; })
            .def_property_readonly("median", [](FWTPTrailer &self) -> uint32_t { return self.median; })
            .def_property_readonly("padding_1", [](FWTPTrailer &self) -> uint32_t { return self.padding_1; })
            .def_property_readonly("padding_2", [](FWTPTrailer &self) -> uint32_t { return self.padding_2; })
            .def_property_readonly("padding_3", [](FWTPTrailer &self) -> uint32_t { return self.padding_3; })
            .def_property_readonly("padding_4", [](FWTPTrailer &self) -> uint32_t { return self.padding_4; })
            ;

        py::class_<FWTP>(m, "FWTP")
          .def("get_n_hits", &FWTP::get_n_hits, py::return_value_policy::reference)
          .def("get_header", &FWTP::get_header, py::return_value_policy::reference)
          .def("get_trailer", &FWTP::get_trailer, py::return_value_policy::reference)
          .def("get_data", &FWTP::get_data, py::return_value_policy::reference)
          ;

        m.def("unpack_fwtps", &unpack_fwtps);

      }

    } // namespace python
  }   // namespace dtpfeedbacktools
} // namespace dunedaq
