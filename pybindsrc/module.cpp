/**
 * @file module.cpp
 *
 * This is part of the DUNE DAQ Software Suite, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iomanip>

#include "dtpfeedbacktools/RawFileReader.hpp"
#include "dtpfeedbacktools/FWTP.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T, typename U> constexpr size_t offset_of(U T::*member)
{
    return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

namespace dunedaq {
namespace dtpfeedbacktools {
namespace python {

std::vector<FWTP> check_fwtps(void *buf, size_t n_blocks, bool safe_mode=true){

}


std::vector<FWTP> unpack_fwtps(void *buf, size_t n_blocks, bool safe_mode=true){

    constexpr uint16_t trl_magic = 0xf00d;

    std::vector<FWTP> fwtps;
    fwtps.reserve(n_blocks/3);


    size_t offset(0), offset_pad1(0), offset_trl(0);

    // Search for the first trailer
    // discard block before it

    // Work in bytes
    auto buf_u8 = static_cast<uint8_t*>(buf);
    size_t buf_size = n_blocks * sizeof(FWTPHeader);

    // Find the first occurence of f00d
    for (; offset_pad1<buf_size; ++offset_pad1) {

      if (((buf_u8[offset_pad1+1]<<8)+buf_u8[offset_pad1]) != trl_magic) {
        continue;
      }

      if (offset_pad1 < offset_of(&FWTPTrailer::padding_1)) {
        continue;
      }

      break;
    }

    // work back the trailer offset
    offset_trl= offset_pad1-offset_of(&FWTPTrailer::padding_1);

    // calculate the binary to mem block phase
    offset = offset_trl % sizeof(FWTPHeader);

    size_t i_trl_0((offset_trl-offset)/sizeof(FWTPHeader));

    FWTPHeader *hdr = static_cast<FWTPHeader *>(buf+offset);
    FWTPTrailer *trl = static_cast<FWTPTrailer *>(buf+offset);

    // Initialize indexes
    size_t i = 0;


    // In safe mode start from the block after the first trailer
    if ( safe_mode ) {
      i = i_trl_0+1;
    }

  // std::cout << "starting from block " << i << " (byte offset " ")" << offset << std::endl;

    size_t i_hdr(i), i_trl(i);

    //std::cout << "n blocks " << n_blocks << std::endl;

    for (; i<n_blocks; ++i) {

      if (trl[i].padding_1 != trl_magic) {
        //std::cout << "Something went terribly wrong..." << std::endl;
        //std::cout << "trl.padding_1 " << trl[i].padding_1 << std::endl;
        continue;
      }
      //std::cout << "iblock " << i << std::endl;
      //std::cout << "trl.padding_1 " << trl[i].padding_1 << std::endl;

      i_trl = i;

      int64_t n_hits = ((int64_t)i_trl-((int64_t)i_hdr+1));
      if (  n_hits < 1 ) {
        std::cout << "Block " << i << " -> " << n_hits << " ~ " <<i_trl << " " << i_hdr << std::endl;
      } else {
        fwtps.push_back(FWTP((void*)(hdr+i_hdr), i_trl-(i_hdr+1)));
      }

      i_hdr = i+1;
    }

    return fwtps;
}

//-----------------
py::dict fwtps_to_arrays(const std::vector<FWTP>& v) {

  size_t n_hits = 0;
  for(const FWTP& tp : v ) {
    n_hits += tp.get_n_hits();
  }
  
  // std::cout << "Detected " << n_hits << " in " <<  v.size() << " FWTPs" << std::endl;

  py::array_t<uint64_t> ts(n_hits);
  py::array_t<uint32_t> crate_no(n_hits), slot_no(n_hits), fiber_no(n_hits), wire_no(n_hits), flags(n_hits), median(n_hits), start_time(n_hits), end_time(n_hits), peak_time(n_hits), peak_adc(n_hits), hit_continue(n_hits), tp_flags(n_hits), sum_adc(n_hits);
  py::array_t<int16_t> accumulator(n_hits);

  auto ts_p = static_cast<uint64_t*>(ts.request().ptr);
  auto crate_no_p = static_cast<uint32_t*>(crate_no.request().ptr);
  auto slot_no_p = static_cast<uint32_t*>(slot_no.request().ptr);
  auto fiber_no_p = static_cast<uint32_t*>(fiber_no.request().ptr);
  auto wire_no_p = static_cast<uint32_t*>(wire_no.request().ptr);
  auto flags_p = static_cast<uint32_t*>(flags.request().ptr);
  auto median_p = static_cast<uint32_t*>(median.request().ptr);
  auto accumulator_p = static_cast<int16_t*>(accumulator.request().ptr);
  auto start_time_p = static_cast<uint32_t*>(start_time.request().ptr);
  auto end_time_p = static_cast<uint32_t*>(end_time.request().ptr);
  auto peak_time_p = static_cast<uint32_t*>(peak_time.request().ptr);
  auto peak_adc_p = static_cast<uint32_t*>(peak_adc.request().ptr);
  auto hit_continue_p = static_cast<uint32_t*>(hit_continue.request().ptr);
  auto tp_flags_p = static_cast<uint32_t*>(tp_flags.request().ptr);
  auto sum_adc_p = static_cast<uint32_t*>(sum_adc.request().ptr);

  size_t i(0);
  for(const FWTP& tp : v ) {
    const auto& tph = tp.get_header();
    const auto& tpt = tp.get_trailer();
    
    for( size_t j(0); j<tp.get_n_hits(); ++j) {

      const auto& tpd = tp.get_data(j);

      ts_p[i] = tph.get_timestamp();
      crate_no_p[i] = tph.crate_no;
      slot_no_p[i] = tph.slot_no;
      fiber_no_p[i] = tph.fiber_no;
      wire_no_p[i] = tph.wire_no;
      flags_p[i] = tph.flags;
      median_p[i] = tpt.median;
      accumulator_p[i] = tpt.accumulator;
      start_time_p[i] = tpd.start_time;
      end_time_p[i] = tpd.end_time;
      peak_time_p[i] = tpd.peak_time;
      peak_adc_p[i] = tpd.peak_adc;
      hit_continue_p[i] = tpd.hit_continue;
      tp_flags_p[i] = tpd.tp_flags;
      sum_adc_p[i] = tpd.sum_adc;
      ++i;
    }

  }

  py::dict fwtp_arrays(
    "ts"_a=ts,
    "crate_no"_a=crate_no,
    "slot_no"_a=slot_no,
    "fiber_no"_a=fiber_no,
    "wire_no"_a=wire_no,
    "flags"_a=flags,
    "median"_a=median,
    "accumulator"_a=accumulator,
    "start_time"_a=start_time,
    "end_time"_a=end_time,
    "peak_time"_a=peak_time,
    "peak_adc"_a=peak_adc,
    "hit_continue"_a=hit_continue,
    "tp_flags"_a=tp_flags,
    "sum_adc"_a=sum_adc
    );


  return fwtp_arrays;
}

py::dict unpack_fwtps_to_arrays(void *buf, size_t n_blocks, bool safe_mode=true) {

    auto fwtps = unpack_fwtps(buf, n_blocks, safe_mode);
    return fwtps_to_arrays(fwtps);
}



PYBIND11_MODULE(_daq_dtpfeedbacktools_py, m)
{
  m.doc() = "c++ implementation of the dunedaq dtp feedbac tools modules"; // optional module docstring

  py::class_<RawDataBlock>(m, "RawDataBlock")
      .def("data", &RawDataBlock::data)
      .def("size", &RawDataBlock::size)
      .def("as_bytes",
        [](RawDataBlock &self) 
        {return py::bytes(self.data(), self.size());},
        py::return_value_policy::reference_internal
        )
      .def(
          "as_capsule",
          [](RawDataBlock &self) -> void*
          { return static_cast<void *>(self.data()); },
          py::return_value_policy::reference_internal
          );

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
      .def_property_readonly("accumulator", [](FWTPTrailer &self) -> int32_t { return self.accumulator; })
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

  m.def("check_fwtps", &check_fwtps, "buf"_a, "n_blocks"_a, "safe_mode"_a = true);
  m.def("unpack_fwtps", &unpack_fwtps, "buf"_a, "n_blocks"_a, "safe_mode"_a = true);
  m.def("fwtps_to_arrays", &fwtps_to_arrays);
  m.def("unpack_fwtps_to_arrays", &unpack_fwtps_to_arrays, "buf"_a, "n_blocks"_a, "safe_mode"_a = true);

}

} // namespace python
}   // namespace dtpfeedbacktools
} // namespace dunedaq
