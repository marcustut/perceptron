#import "template.typ": *
#import "@preview/cetz:0.1.2": canvas, plot, palette, draw

#let legend-item(point, name, style) = {
  draw.content(
    (point.at(0)+2, point.at(1)), (point.at(0)+2, point.at(1)), 
    frame: "rect",
    padding: .3em,
    fill: style.fill,
    stroke: none,
    [],
  )
  draw.content(
    (point.at(0)+2.4, point.at(1)+0.1), (point.at(0)+5.5, point.at(1)),
    [
      #v(.2em)
      #text(name, size: .7em, weight: "bold")
    ]
  )
}

#show: project.with(
  title: "Multi Layer Perceptron",
  authors: (
    (name: "Lee Kai Yang (23205838)", email: "kai.y.lee@ucdconnect.ie"),
  ),
  date: "December 4, 2023",
  font: "CMU Serif",
  monofont: "CMU Typewriter Text",
)

= Introduction

The aim of this paper is to implement a multi layer perceptron (aka Neural Network) without using any external neural network or machine learning libraries. The language I chose is C++, specifically #strong("C++20"). Hence, there is a some manual effort required to compile the program since the it is a static language.

== Compilation instructions 

To compile the project following tools has to be installed on the system:

- *CMake* - build tool for the project
- *libfmt* - a replacement for `std::cout`

These can be installed by downloading the executable on their official website or if you're on Ubuntu or MacOS:

```sh
# Ubuntu
sudo apt update && sudo apt install cmake libfmt-dev

# MacOS
brew install cmake && brew install libfmt
```

After that you can run the following commands to compile the executables

```sh
# Build the library
mkdir build && cd build && cmake .. && cmake --build . && make all

# Run the executables
./examples/xor/perceptron_example_xor
./examples/xor/perceptron_example_sin
./examples/xor/perceptron_example_letter_recognition
```

== Code structure

The source code can be found in the attached zip file and it contains the tests, plots, data, etc. The multi layer perceptron is implemented as C++ library and all tasks accomplished below have their own source code importing the `MultiLayerPerceptron` class. The source code mainly resides in these few folders:

```sh
perceptron
├── include
├── src
├── examples
└── tests
```

Written code for the neural network can be found in `include/perceptron` and `src` whereas `tests` and `examples` houses some unit tests and the executable for the tasks.

= Completed Tasks

== Training a XOR network

The code for this can be found at `examples/xor/xor.cpp`. The code block below is the actual code that construct the network and Figure 1 below shows a visualisation of the network.

```cpp
#define IN_FEATURES 2
#define OUT_FEATURES 1
#define HIDDEN_FEATURES 4
#define MAX_EPOCHS 10000
#define LEARNING_RATE 3.0
#define TARGET_ERROR 0.001
#define BATCH_SIZE 2

auto mlp = perceptron::MultiLayerPerceptron(
        std::vector<perceptron::Layer>{
                perceptron::Layer(IN_FEATURES, HIDDEN_FEATURES, activation),
                perceptron::Layer(HIDDEN_FEATURES, OUT_FEATURES, activation)
        },
        randomizer
);
```

#figure(
  image("images/xor_network.png", width: 40%),
  caption: [XOR Network]
)

The results is as follows:

```sh
Epoch 534: 0.001006418985527181 (error)
Epoch 535: 0.001003637289454922 (error)
Epoch 536: 0.0010008703262003582 (error)
Error is less than target error (0.001). Stopping...
[0, 0]: [[0.027154, ]]
[0, 1]: [[0.968392, ]]
[1, 0]: [[0.967761, ]]
[1, 1]: [[0.0348811, ]]
```

As can be seen although the max epochs is 10000, the network converges much earlier than that with only 536 epochs. In fact, I have also tested running it with 2 to 3 hidden units, the network also converges only that it needs more epochs. Note that, there is a `BATCH_SIZE` here because this is ran using *Stochastic Gradient Descent (SGD)*. The network can also be run without SGD by using the `.train()` method instead of `.SGD()`. 

== Learning $sin(x)$ function

The code for this can be found at `examples/sin/sin.cpp` and the configuration and execution result is as follows: 

```cpp
#define IN_FEATURES 4
#define OUT_FEATURES 1
#define HIDDEN_FEATURES 8
#define MAX_EPOCHS 10000
#define LEARNING_RATE 0.001
#define TARGET_ERROR 0.001
#define BATCH_SIZE 100

auto mlp = perceptron::MultiLayerPerceptron(
        std::vector<perceptron::Layer>{
                perceptron::Layer(IN_FEATURES, HIDDEN_FEATURES, activation),
                perceptron::Layer(HIDDEN_FEATURES, OUT_FEATURES, activation)
        },
        randomizer
);
```

```sh
Epoch 9700: error is 7.881205768825114
Epoch 9800: error is 7.873048266608942
Epoch 9900: error is 7.864983282357822
Epoch 10000: error is 7.857008192932907
Input: [-0.20771906768934767, -0.28155039081825717, 0.4261937947634129, -0.6844652395547632]: Expected [0.9263071737385499], Got [[0.925052, ]]
```

Although this time the network does not achieve the `TARGET_ERROR` within 10000 epochs, it still did quite well a plot is provided below by using the expected values and the predictions from the network, the code for this plot is in `plot.py`.

#figure(
  image("images/sin_predictions.png", width: 85%),
  caption: [Predictions vs targets of sin(x) values]
)

Looking at the figure above, we can see that the network is able to learn the patterns quite well, it almost resembles the original data points. In my opinion, the results is quite satisfactory given how quick the network arrives at this result. From my testing, using *SGD* allows the network to arrive at SSR to local minima surprisingly faster than normal training. The results tabulated below shows a comparison in number of epoch and the error between the two. Note that the actual time taken to arrive at a certain epoch is very similar for the two and the difference is almost negligible.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
  align: horizon,
  [*Normal*], [*SGD*],
  [565.70 (100)],
  [6.99 (100)],
  [249.13 (1100)],
  [5.37 (1100)],
  [32.03 (2100)],
  [5.15 (2100)],
  ),
  caption: [Sum of Squared Residuals (SSR) for SGD vs Normal Training]
)

Note that the number shown above is calculated using _Sum of Squared Residuals (SSR)_ as a loss function and the number inside the parenthesis is the number of epoch. It can be seen that SGD converge much faster and in fact this phenomenon is heavily affected by the `BATCH_SIZE` configuration, I find that smaller batch sizes leads to faster convergence on average.

== Letter recognition

The code for this part can be found in `examples/letter_recognition/letter_recognition.cpp`. The configuration for the network is as follows: 

```cpp
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
```

The first feature of the dataset is a capital letter which is the expected label hence, the network only takes the rest of the 16 features as inputs. As for outputs, it has 26 neurons because there are 26 capital letters from `A-Z`.

The code for data loading, processing and train test split is also included in the executable file. The gist of it is that it reads the file as a vector of strings and match over it using a regular expression. All the 16 features are used to construct the input vector and the first feature is encoded using a vector of 25 zeros and 1 one corresponding to the letter indices. For example, the letter A would be encoded as `[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]`.

Despite trying muliple combinations of activation functions, loss functions, adjusting the number of hidden units and the hyperparameters, the network is not producing a sensible result for the output. I have tried *Sigmoid* at the hidden layer and *Softmax* at the output layer which in theory should work but does not. The outputs that I get are just a vector of values between 0 and 1, not exactly one of them which can not exactly be classified as a letter.


= Conclusion

I chose C++ specifically for this assignment because I know it will be the hardest for me to implement a working neural network as it is much lower level compared to others eg. Python, Julia, etc. It was considerably easy to write the feed-forward part of the network however I struggled a lot in implementing backpropagation. The resources online are mostly Python and the mathematic expressions has a lot of notations and not that straightforward to understand. For a few days I was stuck having a feed-forward only network until I sit down and figure out the math behind backpropagation, after I figure out all the intermediate steps needed for calculating the gradients for the entire network only that I was able to correctly implement it in code.

That said, there's still challenges faced as it involves a lot of matrix multiplication and I often get runtime error due to incorrect matrix dimensions. That said, having it completed now and looking back I really appreciate the people behind libraries such as PyTorch, Tensorflow, numpy, etc. as their work is really no easy tasks and without them writing a neural network without any point of reference would be quite difficult.

I like to highlight the moment where every clicked for me is when it comes to my realisation that neural network is nothing more than just an abstract data structure, like a tree which can be represented in memory using adjacency lists (a.k.a matrix and vectors). Ever since then, it made much more sense while writing the code for it myself. In terms of discovery, it is also the first time I ever wrote *Stochastic Gradient Descent (SGD)*  without any libraries and only then I realise how amazing it is that such big improvements only require such less code.