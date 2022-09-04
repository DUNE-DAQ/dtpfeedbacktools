from ._daq_dtpfeedbacktools_py import *


import ctypes

class FWTPHeader(ctypes.Structure):
    _fields_ = [
        ('wire_no', ctypes.c_uint32, 8),
        ('slot_no', ctypes.c_uint32, 3),
        ('unused_0', ctypes.c_uint32, 1),
        ('flags', ctypes.c_uint32, 4),
        ('crate_no', ctypes.c_uint32, 10),
        ('fiber_no', ctypes.c_uint32, 6),
        ('timestamp_1', ctypes.c_uint32, 16),
        ('timestamp_2', ctypes.c_uint32, 16),
        ('timestamp_3', ctypes.c_uint32, 16),
        ('timestamp_4', ctypes.c_uint32, 16),
    ]

class FWTPData(ctypes.Structure):
    _fields_ = [
        ('end_time', ctypes.c_uint32, 16),
        ('start_time', ctypes.c_uint32, 16),
        ('peak_time', ctypes.c_uint32, 16),
        ('peak_adc', ctypes.c_uint32, 16),
        ('hit_continue', ctypes.c_uint32, 1),
        ('tp_flags', ctypes.c_uint32, 15),
        ('sum_adc', ctypes.c_uint32, 16),
    ]
class FWTPTrailer(ctypes.Structure):
    _fields_ = [
        ('accumulator', ctypes.c_int32, 16),
        ('median', ctypes.c_uint32, 16),
        ('m_padding_1', ctypes.c_uint32, 16),
        ('m_padding_2', ctypes.c_uint32, 16),
        ('m_padding_3', ctypes.c_uint32, 16),
        ('m_padding_4', ctypes.c_uint32, 16),
    ]