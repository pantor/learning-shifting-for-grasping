# - Try to find EnsensoSDK
# Once done this will define
#  ENSENSO_FOUND
#  ENSENSO_INCLUDE_DIRS
#  ENSENSO_LIBRARIES
#  ENSENSO_DEFINITIONS

find_package(PkgConfig)
pkg_check_modules(PC_ENSENSO QUIET ensenso-nxlib)
set(ENSENSO_DEFINITIONS ${PC_ENSENSO_CFLAGS_OTHER})

find_path(ENSENSO_INCLUDE_DIR nxLib.h
          HINTS ${PC_ENSENSO_INCLUDEDIR} ${PC_ENSENSO_INCLUDE_DIRS})

find_library(ENSENSO_LIBRARY NAMES NxLib32 NxLib64
             HINTS ${PC_ENSENSO_LIBDIR} ${PC_ENSENSO_LIBRARY_DIRS})

set(ENSENSO_LIBRARIES ${ENSENSO_LIBRARY})
set(ENSENSO_INCLUDE_DIRS ${ENSENSO_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ENSENSO_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Ensenso DEFAULT_MSG
                                  ENSENSO_LIBRARY ENSENSO_INCLUDE_DIR)

mark_as_advanced(ENSENSO_INCLUDE_DIR ENSENSO_LIBRARY)
