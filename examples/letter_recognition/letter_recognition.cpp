#include "perceptron/perceptron.hpp"

#define IN_FEATURES 16
#define OUT_FEATURES 26
#define HIDDEN_FEATURES 16

#define MAX_EPOCHS 100
#define LEARNING_RATE 0.001
#define TARGET_ERROR 0.001
#define BATCH_SIZE 100

#define DATASET_PATH "letter-recognition.data"
#define DATASET_REGEX "([A-Z]),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+),(\\d+)"
#define TRAIN_DATA_PCT 0.8


auto encode_output_feature = [](const std::string &letter) {
    auto idx = static_cast<size_t>(letter[0] - 'A');
    auto output = std::vector<Scalar>(26, 0);
    output[idx] = 1;
    return output;
};

auto decode_output_feature = [](const std::vector<Scalar> &outputs) {
    for (int i = 0; i < outputs.size(); i++)
        if (outputs[i] == 1) return std::string(1, static_cast<char>(i + 'A'));
    return std::string("?");
};

auto read_data() {
    auto read_file_lines = [](const std::string &path) {
        std::ifstream file(path);
        std::string line;
        std::vector<std::string> lines;

        if (!file.is_open()) throw std::runtime_error("Failed to open file '" + path + "'");

        while (getline(file, line)) lines.push_back(line);

        return lines;
    };

    std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> train_data;
    std::regex regex(DATASET_REGEX);

    for (auto &line: read_file_lines(DATASET_PATH)) {
        std::smatch match;
        std::regex_match(line, match, regex);
        auto letter = match[1].str();
        auto features = std::vector<Scalar>{
                std::stod(match[2].str()),
                std::stod(match[3].str()),
                std::stod(match[4].str()),
                std::stod(match[5].str()),
                std::stod(match[6].str()),
                std::stod(match[7].str()),
                std::stod(match[8].str()),
                std::stod(match[9].str()),
                std::stod(match[10].str()),
                std::stod(match[11].str()),
                std::stod(match[12].str()),
                std::stod(match[13].str()),
                std::stod(match[14].str()),
                std::stod(match[15].str()),
                std::stod(match[16].str()),
                std::stod(match[17].str()),
        };
        auto target = encode_output_feature(letter);
        train_data.emplace_back(features, target);
    }

    return train_data;
}

auto train_test_split(std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> data,
                      const double train_data_pct) {
    // Shuffle the data
    auto rng = std::default_random_engine(std::random_device()());
    std::shuffle(data.begin(), data.end(), rng);

    std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> train_data, test_data;

    for (auto &item: data)
        if (train_data.size() < train_data_pct * data.size()) train_data.push_back(item);
        else test_data.push_back(item);

    return std::make_pair(train_data, test_data);
}

int main() {
    auto data = read_data();
    fmt::println("Loaded {} sets of data from '{}'", data.size(), DATASET_PATH);
    fmt::println("Splitting data according to {}/{} train test split", TRAIN_DATA_PCT * 100,
                 round((1 - TRAIN_DATA_PCT) * 100));
    auto [train_data, test_data] = train_test_split(std::move(data), TRAIN_DATA_PCT);
    fmt::println("There are {} train data and {} test data.", train_data.size(), test_data.size());

    auto randomizer = perceptron::random::Xavier<Scalar>(IN_FEATURES, OUT_FEATURES);
    auto sigmoid = perceptron::activation::Sigmoid<Scalar>();
    auto relu = perceptron::activation::ReLU<Scalar>();
    auto mlp = perceptron::MultiLayerPerceptron(
            std::vector<perceptron::Layer>{
                    perceptron::Layer(IN_FEATURES, HIDDEN_FEATURES, sigmoid),
                    perceptron::Layer(HIDDEN_FEATURES, OUT_FEATURES, relu)
            },
            randomizer
    );

    auto losses = nc::NdArray<Scalar>();
    auto accuracies = nc::NdArray<Scalar>();

    auto temp = std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>>(
            train_data.begin() + train_data.size() - 100, train_data.end());
    auto loss = perceptron::loss::CrossEntropy<Scalar>();
    auto on_epoch_handler = [&](const int epoch, const perceptron::loss::Loss<Scalar> &loss) {
        auto evaluation = mlp.evaluate(temp, loss);
        if (evaluation.error < TARGET_ERROR) {
            fmt::println("Error is less than target error ({}). Stopping...", TARGET_ERROR);
            return true;
        }
        fmt::println("Epoch {}: error is {}, accuracy is {}", epoch, evaluation.error, evaluation.accuracy);
//        fmt::println("Epoch {}", epoch);
        return false;
    };

    fmt::println("\nStarted training with learning rate {}, max epochs {} and batch size {}...", LEARNING_RATE,
                 MAX_EPOCHS, BATCH_SIZE);

    mlp.train(train_data, loss, MAX_EPOCHS, LEARNING_RATE, on_epoch_handler);
//    mlp.SGD(train_data, loss, MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE, on_epoch_handler);

//    auto targets = nc::NdArray<Scalar>();
//    auto predictions = nc::NdArray<Scalar>();

    for (auto &[input, target]: test_data) {
        auto prediction = mlp.predict(input);
//        fmt::println("Input: {}: Expected {}, Got {}", input, decode_output_feature(target),
//                     decode_output_feature(prediction.toStlVector()));
        fmt::println("Input: {}: Expected {}, Got {}", input, target, prediction);
//        targets = nc::append(targets, nc::NdArray(target));
//        predictions = nc::append(predictions, prediction);
    }

//    nc::tofile(targets, EXPORT_TARGETS_PATH, ',');
//    nc::tofile(predictions, EXPORT_PREDICTIONS_PATH, ',');

    return 0;
}
