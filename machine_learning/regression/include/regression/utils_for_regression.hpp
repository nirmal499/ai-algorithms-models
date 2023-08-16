#ifndef REGRESSION_UTILS_H
#define REGRESSION_UTILS_H

#include <cmath>
#include <vector>
#include <tuple>
#include <array>
#include <algorithm>
#include <iomanip>
#include <random>
#include <ctime>
#include <cmath>
#include <common_utils/for_plot.hpp>

namespace regression_utils{
	// std::random_device dev;
	// std::mt19937 rng(dev());
	// // std::uniform_int_distribution<std::mt19937::result_type> dist1(0.0, 1.0); // It gives either 0 or 1

	void initialize(std::vector<double> features, std::vector<double> labels);

	std::tuple<double, double> square_trick_for_linear_regression(double a, double b, double num_rooms, double price, double learning_rate);
	std::tuple<double, double, double> square_trick_for_quadratic_regression(double a, double b, double c, double num_rooms, double price, double learning_rate);

	// Linear equation: ax + b
	std::tuple<double,double> linear_regression(std::vector<double> features, std::vector<double> labels,
											double learning_rate, unsigned int epochs);
	// Linear equation: ax^2 + bx^1 + c
	std::tuple<double,double,double> quadratic_regression(std::vector<double> features, std::vector<double> labels,
											double learning_rate, unsigned int epochs);
}

#endif