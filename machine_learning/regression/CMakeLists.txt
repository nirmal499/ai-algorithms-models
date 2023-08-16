add_library(regression
	src/utils_for_regression.cpp
)

target_include_directories(regression
	PUBLIC include
)

target_link_libraries(regression
	PRIVATE common_utils
)