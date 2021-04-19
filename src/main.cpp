#include <cstdlib>
#include <cxxopts.hpp>

int main(int argc, char* argv[]) {
	cxxopts::Options options("VOILA compiler", "");

	options.add_options()
		("h,help", "Show help")
		;

	try {
		auto cmd = options.parse(argc, argv);

		if (cmd.count("h")) {
			std::cout << options.help({"", "Group"}) << std::endl;
			exit(0);
		}
	} catch (const cxxopts::OptionException& e) {
		std::cerr << "error parsing options: " << e.what() << std::endl;
		std::cout << options.help({"", "Group"}) << std::endl;
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}