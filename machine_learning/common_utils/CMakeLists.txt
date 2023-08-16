add_subdirectory(external/matplotplusplus)

add_library(common_utils
	src/for_plot.cpp
)

target_include_directories(common_utils
	PUBLIC include
)

# Linking will also include the header files in our hpp files
target_link_libraries(common_utils
	# PRIVATE matplot; Giving ERROR that "#include <matplot/matplot.h>" does not exists
	# PUBLIC matplot
	PRIVATE matplot
)

#[[ Inorder for the "PRIVATE matplot" to work I shifted the "#include <matplot/matplot.h>" from for_plot.hpp to for_plot.cpp ]]
