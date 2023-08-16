#include <cmath>
#include <matplot/matplot.h>
#include <vector>

int main(void)
{
	std::vector<double> x = matplot::linspace(0,180,7);
	std::vector<double>y = {0.8, 0.9, 0.1, 0.9, 0.6, 0.1, 0.3};

	matplot::plot(x,y);
	matplot::hold(matplot::on);
	matplot::scatter(x,y);
	matplot::hold(matplot::off);

	matplot::title("Time Plot");
	matplot::xlabel("Time");
	matplot::yrange({0,1});

	matplot::xticks({0, 30, 60, 90, 120, 150, 180});
	matplot::xticklabels({"00:00s", "30:00", "01:00", "01:30", "02:00", "02:30", "03:00"});

	matplot::show();


	return 0;
}