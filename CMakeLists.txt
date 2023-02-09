cmake_minimum_required(VERSION 3.12)
project(dtpfeedbacktools VERSION 1.0.0)

find_package(daq-cmake REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(CLI11 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

daq_setup_environment()


##############################################################################
# Main library

daq_add_library(*.cpp LINK_LIBRARIES fmt::fmt)


##############################################################################
daq_add_python_bindings(*.cpp LINK_LIBRARIES ${PROJECT_NAME})

# #############################################################################
# Plugins

# #############################################################################
# Integration tests
daq_add_application(dtpfb_test_reader dtpfb_test_reader.cxx TEST LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(dtpft_check_tp_dump dtpft_check_tp_dump.cxx TEST LINK_LIBRARIES CLI11::CLI11 ${PROJECT_NAME} nlohmann_json)
daq_add_application(dtpft_test_unpack_fwtps dtpft_test_unpack_fwtps.cxx TEST LINK_LIBRARIES CLI11::CLI11 ${PROJECT_NAME} nlohmann_json fmt::fmt)

# #############################################################################
# Unit Tests
# daq_add_unit_test(WIBFrame_test             LINK_LIBRARIES ${PROJECT_NAME})
# daq_add_unit_test(WIB2Frame_test            LINK_LIBRARIES detdataformats)
# daq_add_unit_test(TDE16Frame_test           LINK_LIBRARIES detdataformats)
# daq_add_unit_test(TDE12Frame_test           LINK_LIBRARIES detdataformats)
# daq_add_unit_test(DetID_test                LINK_LIBRARIES detdataformats)
# #############################################################################


daq_install()
