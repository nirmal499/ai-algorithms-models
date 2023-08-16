#include <cmath>
#include <matplot/matplot.h>
#include <vector>

int main(void)
{
	int starting = 1;
	int ending = 10;
	std::vector<double> x = matplot::linspace(starting, ending, 1000);

	for(const auto& element: x){
		std::cout << element << "\n";
	}

	return 0;
}