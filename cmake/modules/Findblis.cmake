# - Find BLIS library
#
# This module sets the following variables:
#  BLIS_FOUND - set to true if a library implementing CBLAS interface is found.
#  BLIS_INCLUDE_DIR - path to include dir.
#  BLIS_LIB - list of libraries for BLIS.
#
#
 
SET(BLIS_INCLUDE_SEARCH_PATHS
   ${CMAKE_SOURCE_DIR}/blis/include/firestorm/
 )
 
 SET(BLIS_LIB_SEARCH_PATHS
    ${CMAKE_SOURCE_DIR}/blis/lib/firestorm/
 )
 
 FIND_PATH(BLIS_INCLUDE_DIR NAMES blis.h
           PATHS ${BLIS_INCLUDE_SEARCH_PATHS})
 #    Check include files
 IF(NOT BLIS_INCLUDE_DIR)
        SET(BLIS_FOUND OFF)
        MESSAGE(WARNING "Could not find BLIS include.")
        RETURN()
 ENDIF()
 
 
 FIND_LIBRARY(BLIS_LIB NAMES blis PATHS ${BLIS_LIB_SEARCH_PATHS})
 #    Check libraries
 IF(NOT BLIS_LIB)
        SET(BLIS_FOUND OFF)
        MESSAGE(WARNING "Could not find BLIS lib.")
        RETURN()
 ENDIF()
 
 SET(BLIS_FOUND ON)
 
 IF(BLIS_FOUND)
         IF(NOT BLIS_FIND_QUIETLY)
                 MESSAGE(STATUS "Found BLIS libraries: ${BLIS_LIB}")
                 MESSAGE(STATUS "Found BLIS include: ${BLIS_INCLUDE_DIR}")
         ENDIF()
 ELSE()
         MESSAGE(FATAL_ERROR "Could not find BLIS")
 ENDIF()
 
 INCLUDE(FindPackageHandleStandardArgs)
 FIND_PACKAGE_HANDLE_STANDARD_ARGS(BLIS DEFAULT_MSG BLIS_INCLUDE_DIR BLIS_LIB)
 
 MARK_AS_ADVANCED(
         BLIS_INCLUDE_DIR
         BLIS_LIB
         blis
 )