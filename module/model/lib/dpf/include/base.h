#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <limits>
#include <deque>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <iterator> 
#include <numeric>
#include <cmath>
#include <cstddef> 
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include <cfloat>

#ifdef DEBUG
#define runtime_assert(x) {if (!(x)) { printf("error!\n"); printf(#x); assert(#x); abort(); }}
#else
#define runtime_assert(x) {}
#endif

#define DISC_EPS 1e-6


#define USE_PRUNE true
#define USE_SIM_BOUND false
#define IMB_SORT (USE_PRUNE && true)
#define USE_CACHE true
#define SPECIAL_D2 true
#define BETTER_MERGE true
#define POST_PARETO true
#define SMART_DOM true
