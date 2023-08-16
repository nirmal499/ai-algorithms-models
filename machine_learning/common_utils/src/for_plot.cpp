#include <common_utils/for_plot.hpp>
#include <matplot/matplot.h>

namespace common_utils_for_plot{

	// Linear line equation : ax^1 + b
	void draw_linear_line(double slope_of_x1, double intercept, std::array<float, 3> color, 
						  float linewidth = 0.7, unsigned int starting = 0, unsigned int ending = 8){
		std::vector<double> x = matplot::linspace(starting, ending, 1000);

		std::vector<double> y = matplot::transform(x, [slope_of_x1, intercept](auto x){
			return (slope_of_x1 * x) + intercept;
		});

		auto p = matplot::plot(x, y);
		p->line_width(linewidth);
	}

	// Linear line equation : ax^2 + bx^1 + c
	void draw_quadratic_line(double slope_of_x2, double slope_of_x1, double intercept, std::array<float, 3> color, 
						  float linewidth = 0.7, unsigned int starting = 0, unsigned int ending = 8){
		std::vector<double> x = matplot::linspace(starting, ending, 1000);

		std::vector<double> y = matplot::transform(x, [slope_of_x2, slope_of_x1, intercept](auto x){
			return (slope_of_x2 * std::pow(x,2)) + (slope_of_x1 * x) + intercept;
		});

		auto p = matplot::plot(x, y);
		p->line_width(linewidth);

	}

	void plot_points(std::vector<double> features, std::vector<double> labels, std::string xlabel = "", std::string ylabel = ""){
		matplot::scatter(features, labels);

		matplot::xlabel(xlabel);
		matplot::ylabel(ylabel);

		// matplot::xrange({0,8});
		// matplot::yrange({0,500});

		// matplot::xrange({ 0 , max_value_in_features + 10 });
		// matplot::yrange({ 0 , max_value_in_labels + 10 });
	}

	void matplot_on(){
		matplot::hold(matplot::on);
	}
	void matplot_off(){
		matplot::hold(matplot::off);
	}
}