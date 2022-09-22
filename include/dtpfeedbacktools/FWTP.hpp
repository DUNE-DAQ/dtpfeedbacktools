#ifndef DUNEDAQ_DTPFEEDBACKTOOLS_FWTP_HPP_
#define DUNEDAQ_DTPFEEDBACKTOOLS_FWTP_HPP_

#include <stdint.h>

namespace dunedaq
{
  namespace dtpfeedbacktools
  {
    struct FWTPHeader
    {
      uint32_t wire_no : 8, slot_no : 4, flags : 4, crate_no : 10, fiber_no : 6;
      uint32_t timestamp_1;
      uint32_t timestamp_2;

      uint64_t get_timestamp() const // NOLINT(build/unsigned)
      {
        uint64_t timestamp = (timestamp_1 & 0xFFFF0000) >> 16;
        timestamp += static_cast<int64_t>(timestamp_1 & 0xFFFF) << 16;
        timestamp += static_cast<int64_t>(timestamp_2 & 0xFFFF0000) << 16;
        timestamp += static_cast<int64_t>(timestamp_2 & 0xFFFF) << 48;
        return timestamp;
      }
    };

    struct FWTPData
    {
      // This struct contains three words of TP values that form the main repeating
      // pattern in the TP block.
      uint16_t end_time, start_time;
      uint16_t peak_time, peak_adc;
      uint16_t hit_continue : 1, tp_flags : 15, sum_adc;
    };

    struct FWTPTrailer
    {
      int16_t  accumulator;
      uint16_t median;
      uint16_t padding_1, padding_2;
      uint16_t padding_3, padding_4;
    };

    class FWTP
    {
    private:
      /* data */
      size_t m_n_hits;
      FWTPHeader *m_hdr;
      FWTPData *m_data;
      FWTPTrailer *m_trl;

    public:
      FWTP(void *ptr, size_t n_hits);
      ~FWTP();

      size_t get_n_hits() const { return m_n_hits; }
      const FWTPHeader &get_header() const { return *m_hdr; }
      const FWTPTrailer &get_trailer() const { return *m_trl; }
      const FWTPData &get_data(size_t i) const
      {
        if (i > m_n_hits)
        {
          throw std::out_of_range("FTP data index out of range");
        }

        return *(m_data + i);
      }
    };

    FWTP::FWTP(void *ptr, size_t n_hits) : m_n_hits(n_hits),
                                           m_hdr(static_cast<FWTPHeader *>(ptr)),
                                           m_data(static_cast<FWTPData *>((void *)m_hdr + sizeof(FWTPHeader))),
                                           m_trl(static_cast<FWTPTrailer *>((void *)m_data + (sizeof(FWTPData) * m_n_hits)))
    {
    }
    FWTP::~FWTP()
    {
    }

  } // namespace dtpfeedbacktools
} // namespace dunedaq

#endif /* DUNEDAQ_DTPFEEDBACKTOOLS_FWTP_HPP_ */