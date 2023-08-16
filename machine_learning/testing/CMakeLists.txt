# add_subdirectory(external/matplotplusplus/matplotplusplus)

add_executable(test
	matplot_testing/test.cpp
	# matplot_testing/test1.cpp

	# regression_testing/linear_regression.cpp
	# regression_testing/quadratic_regression.cpp
)

target_link_libraries(test
	PRIVATE matplot
	# PRIVATE common_utils
	# PRIVATE regression
)