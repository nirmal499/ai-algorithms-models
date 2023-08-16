#ifndef COMMON_UTILS_FOR_PLOT
#define COMMON_UTILS_FOR_PLOT

#include <vector>
#include <array>
#include <string>

namespace common_utils_for_plot{
	// Linear line equation : ax + b
	void draw_linear_line(double slope_of_x1, double intercept, std::array<float, 3> color, 
					  float linewidth, unsigned int starting, unsigned int ending);

	// Linear line equation : ax^2 + bx^1 + c
	void draw_quadratic_line(double slope_of_x2, double slope_of_x1, double intercept, std::array<float, 3> color, 
					  float linewidth, unsigned int starting, unsigned int ending);

	void plot_points(std::vector<double> features, std::vector<double> labels, std::string xlabel, std::string ylabel);

	void matplot_on();
	void matplot_off();
}

#endif