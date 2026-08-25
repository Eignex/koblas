/*
 * HiGHS generates this header from CMake. HFactor and the files it links against read only the version
 * macros and the integer width, so the build supplies it directly rather than running HiGHS's own
 * configuration. HIGHSINT64 stays undefined: HighsInt is then `int`, which is the width koblas's own CSC
 * arrays already carry, so index arrays cross the seam without a widening copy.
 */
#ifndef HCONFIG_H_
#define HCONFIG_H_

#define HIGHS_GITHASH "koblas"
#define HIGHS_VERSION_MAJOR 1
#define HIGHS_VERSION_MINOR 15
#define HIGHS_VERSION_PATCH 1

#endif /* HCONFIG_H_ */
