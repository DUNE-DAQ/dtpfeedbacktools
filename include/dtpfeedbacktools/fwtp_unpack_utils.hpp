#ifndef DUNEDAQ_DTPFEEDBACKTOOLS_INCLUDE_FWTP_UNPACK_UTILS_HPP_
#define DUNEDAQ_DTPFEEDBACKTOOLS_INCLUDE_FWTP_UNPACK_UTILS_HPP_

#include "dtpfeedbacktools/FWTP.hpp"


namespace dunedaq {
namespace dtpfeedbacktools {

/**
 *
 **/
std::vector<FWTP> check_fwtps(void *buf, size_t n_blocks, bool safe_mode=true);

/**
 *
 **/
// std::vector<FWTP> unpack_fwtps_leg(void *buf, size_t n_blocks, bool safe_mode=true);
std::vector<FWTP> unpack_fwtps(void *buf, size_t n_blocks, bool safe_mode=true);


} // namespace dtpfeedbacktools
} // namespace dunedaq

#endif /* DUNEDAQ_DTPFEEDBACKTOOLS_INCLUDE_FWTP_UNPACK_UTILS_HPP_ */
