#include <utils.hpp>

int main(void)
{
	
	std::vector<double> features = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	std::vector<double> labels = { 450, 500, 600, 800, 1100, 1500, 2000, 3000, 5000, 10000 };

	utils::initialize(features, labels);

	// ax^2 + bx^1 + c
	// auto [slope_of_x2, slope_of_x1, intercept] = utils::quadratic_regression(features, labels, 0.0001, 10000);
	auto [slope_of_x2, slope_of_x1, intercept] = utils::quadratic_regression(features, labels, 0.0001, 10000);

	std::cout << std::setprecision(15);
	std::cout << "Slope of x^2: " << slope_of_x2 
		<< "\nSlope of x^1: " << slope_of_x1 
		<< "\nIntercept: " << intercept << "\n";

	matplot::show();

	return 0;
}