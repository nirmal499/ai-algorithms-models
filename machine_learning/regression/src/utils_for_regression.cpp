#include <regression/utils_for_regression.hpp>

namespace regression_utils{

	std::random_device dev;
	std::mt19937 rng(dev());
	// std::uniform_int_distribution<std::mt19937::result_type> dist1(0.0, 1.0); // It gives either 0 or 1

	double min_value_in_features{};
	double max_value_in_features{};
	double min_value_in_labels{};
	double max_value_in_labels{};

	void initialize(std::vector<double> features, std::vector<double> labels){

		/**
		 * srand(time(0));
		 * Function call at global scope; impossible. Move those two lines to the beginning of main() 
		 * and the problem should be solved.
		 */
		srand((unsigned)time(NULL));

		// global variables
		regression_utils::min_value_in_features = *std::min_element(features.begin(), features.end());
		regression_utils::max_value_in_features = *std::max_element(features.begin(), features.end());

		// regression_utils::min_value_in_labels = *std::min_element(labels.begin(), labels.end());
		// regression_utils::max_value_in_labels = *std::max_element(labels.begin(), labels.end());

	}

	// Quadratic ax^1 + b
	std::tuple<double, double> square_trick_for_linear_regression(double a, double b, double num_rooms, double price, double learning_rate){
		double predicted_price = (a * num_rooms) + b;
		a += learning_rate * num_rooms * (price - predicted_price);
		b += learning_rate * (price - predicted_price);

		return std::make_tuple(a,b);
	}

	// Quadratic ax^2 + bx^1 + c
	std::tuple<double, double, double> square_trick_for_quadratic_regression(double a, double b, double c, double num_rooms, double price, double learning_rate){
		double num_rooms_pow_2 = std::pow(num_rooms,2);

		double predicted_price = (a * num_rooms_pow_2) + (b * num_rooms) + c;
		a += learning_rate * (num_rooms_pow_2 * (price - predicted_price));
		b += learning_rate * (num_rooms * (price - predicted_price));
		c += learning_rate * (price - predicted_price);

		return std::make_tuple(a,b,c);
	}

	// Linear equation: ax + b
	std::tuple<double,double> linear_regression(std::vector<double> features, std::vector<double> labels,
												double learning_rate = 0.01, unsigned int epochs = 1000){
		
		// Generating random number between 0 and 1 for "a" and "b"
		double a = (double)rand()/RAND_MAX;
		double b = (double)rand()/RAND_MAX;

		// std::cout << std::setprecision(15) << a << " " << b << "\n";

		std::uniform_int_distribution<std::mt19937::result_type> dist2(0, (features.size() - 1));

		for(unsigned int i = 0; i < epochs; ++i){
			double predictions = (a * features[0]) + b;

			unsigned int idx = dist2(rng);
			// std::cout << idx << "\n";
			double num_rooms = features[idx];
			double price = labels[idx];

			auto [temp_a, temp_b] = square_trick_for_linear_regression(a, b, num_rooms, price, learning_rate);
			a = temp_a;
			b = temp_b;
		}

		common_utils_for_plot::draw_linear_line(a, b, {0.83, 0.14, 0.14}, 0.8, min_value_in_features, max_value_in_features);

		common_utils_for_plot::matplot_on();
		common_utils_for_plot::plot_points(features, labels, "No.of Rooms", "Prices");
		common_utils_for_plot::matplot_off();

		// std::cout << "Price per room: " << std::setprecision (15) << a << "\n";
		// std::cout << "Base price: " << b << "\n";

		// matplot::show();

		return std::make_tuple(a,b);

	}

	// Quadratic equation: ax^2 + bx^1 + c
	std::tuple<double,double,double> quadratic_regression(std::vector<double> features, std::vector<double> labels,
												double learning_rate = 0.01, unsigned int epochs = 1000){
		
		// Generating random number between 0 and 1 for "a" and "b"
		double a = (double)rand()/RAND_MAX;
		double b = (double)rand()/RAND_MAX;
		double c = (double)rand()/RAND_MAX;

		// std::cout <<  a << " " << b << " " << c << "\n";

		// std::cout << std::setprecision(15) << a << " " << b << "\n";
		std::uniform_int_distribution<std::mt19937::result_type> dist2(0, (features.size() - 1));
		
		double features_0_pow_2 = std::pow(features[0],2);

		for(unsigned int i = 0; i < epochs; ++i){
			double predictions = (a * features_0_pow_2) + (b * features[0]) + c;

			unsigned int idx = dist2(rng);
			// std::cout << idx << "\n";
			double num_rooms = features[idx];
			double price = labels[idx];

			auto [temp_a, temp_b, temp_c] = square_trick_for_quadratic_regression(a, b, c, num_rooms, price, learning_rate);

			a = temp_a;
			b = temp_b;
			c = temp_c;

			// std::cout << a << " " << b << " " << c << "\n";
		}

		common_utils_for_plot::draw_quadratic_line(a, b, c, {0.83, 0.14, 0.14}, 0.7, min_value_in_features, max_value_in_features);

		common_utils_for_plot::matplot_on();
		common_utils_for_plot::plot_points(features, labels, "No.of Rooms", "Prices");
		common_utils_for_plot::matplot_off();

		// std::cout << "Price per room: " << std::setprecision (15) << a << "\n";
		// std::cout << "Base price: " << b << "\n";

		// matplot::show();

		return std::make_tuple(a,b,c);

	}
}