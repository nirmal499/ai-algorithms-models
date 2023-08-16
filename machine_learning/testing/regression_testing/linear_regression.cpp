#include <utils.hpp>

int main(void)
{	
	std::vector<double> features = { 1,2,3,5,6,7 };
	std::vector<double> labels = { 155, 197, 244, 356, 407, 448};

	utils::initialize(features, labels);

	auto [slope_of_x, intercept] = utils::linear_regression(features, labels, 0.01, 10000);

	std::cout << std::setprecision(15);
	std::cout << "Slope of x: " << slope_of_x << "\nIntercept: " << intercept << "\n";

	matplot::show();

	return 0;
}