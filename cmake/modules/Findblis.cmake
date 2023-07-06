# - Find BLIS library
#
# This module sets the following variables:
#  BLIS_FOUND - set to true if a library implementing CBLAS interface is found.
#  BLIS_INCLUDE_DIR - path to include dir.
#  BLIS_LIB - list of libraries for BLIS.
#
#

SET(BLIS_INCLUDE_SEARCH_PATHS
   ${CMAKE_SOURCE_DIR}/blis/include/*/
 )
 
 SET(BLIS_LIB_SEARCH_PATHS
    ${CMAKE_SOURCE_DIR}/blis/lib/*/
 )
 
 FIND_PATH(COMET_BLIS_INCLUDE_DIR NAMES blis.h
           PATHS ${BLIS_INCLUDE_SEARCH_PATHS})
 #    Check include files
 IF(NOT COMET_BLIS_INCLUDE_DIR)
        SET(BLIS_FOUND OFF)
        MESSAGE(WARNING "Could not find BLIS include.")
        RETURN()
 ENDIF()
 
 
 FIND_LIBRARY(COMET_BLIS_LIB NAMES blis PATHS ${BLIS_LIB_SEARCH_PATHS})
 #    Check libraries
 IF(NOT COMET_BLIS_LIB)
        SET(BLIS_FOUND OFF)
        MESSAGE(WARNING "Could not find BLIS lib.")
        RETURN()
 ENDIF()
 
 SET(BLIS_FOUND ON)
 
 IF(BLIS_FOUND)
         IF(NOT BLIS_FIND_QUIETLY)
                 MESSAGE(STATUS "Found BLIS libraries: ${COMET_BLIS_LIB}")
                 MESSAGE(STATUS "Found BLIS include: ${COMET_BLIS_INCLUDE_DIR}")
         ENDIF()
 ELSE()
         MESSAGE(FATAL_ERROR "Could not find BLIS")
 ENDIF()
 
 INCLUDE(FindPackageHandleStandardArgs)
 FIND_PACKAGE_HANDLE_STANDARD_ARGS(BLIS DEFAULT_MSG COMET_BLIS_INCLUDE_DIR COMET_BLIS_LIB)
 
 MARK_AS_ADVANCED(
         COMET_BLIS_INCLUDE_DIR
         COMET_BLIS_LIB
         blis
 )