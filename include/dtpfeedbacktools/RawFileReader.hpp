
#ifndef DUNEDAQ_DTPFEEDBACKTOOLS_RAWFILEREADER_HPP_
#define DUNEDAQ_DTPFEEDBACKTOOLS_RAWFILEREADER_HPP_

#include <string>
#include <memory>
#include <fstream>

namespace dunedaq
{
  namespace dtpfeedbacktools
  {

    class RawDataBlock
    {
    public:
      RawDataBlock(size_t size) : m_size(size)
      {
        m_data = new char[size];
      }
      ~RawDataBlock()
      {
        delete[] m_data;
      }

      char* data() { return m_data; }
      size_t size() { return m_size; }

    private:
      char *m_data;
      size_t m_size;

      friend class RawFileReader;
    };

    class RawFileReader
    {
    public:
      RawFileReader(const std::string &path);
      ~RawFileReader();

      size_t get_size() { return m_size; }
      std::unique_ptr<RawDataBlock> read_block(size_t size, size_t offset = 0);

    private:
      std::string m_path;
      size_t m_size;

      std::ifstream m_file;
    };

  }
}

#endif /* DUNEDAQ_DTPFEEDBACKTOOLS_RAWFILEREADER_HPP_ */